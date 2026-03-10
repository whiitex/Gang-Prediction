"""
Demographics-based KYC Assigner

Assigns realistic KYC attributes (age, salary, balance, city) to accounts based on:
1. Demographic statistics (e.g., SCB data)
2. Structural position in the graph

Key insight: balance is derived from salary + structural activity, not random.
This creates meaningful correlations for ML detection.

City assignment creates geographic clustering by propagating city labels
through the network from seed nodes, so nearby accounts share cities.

Demographics CSV Schema
-----------------------
Required columns:
    - population size: Number of people in this age group (used for sampling weights)
    - average year income (tkr): Mean yearly income in thousands of SEK
    - median year income (tkr): Median yearly income in thousands of SEK

Age format (one of):
    Option 1 - Single-year ages:
        - age: Integer age (e.g., 16, 17, 18, ...)

    Option 2 - Age bands:
        - age_low: Lower bound of age band (inclusive)
        - age_high: Upper bound of age band (inclusive)
        Age is sampled uniformly within the band.

Example (single-year format):
    age, average year income (tkr), median year income (tkr), population size
    16, 5.7, 0.0, 118238.0
    17, 11.5, 5.3, 117938.0
    ...

Example (age band format):
    age_low, age_high, average year income (tkr), median year income (tkr), population size
    16, 19, 28.4, 18.2, 467122.5
    20, 24, 188.7, 165.4, 579587.5
    ...
"""

import numpy as np
import pandas as pd
import networkx as nx
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DemographicsAssigner:
    """
    Assigns demographically-realistic KYC attributes to graph nodes.

    Uses Swedish Central Bureau of Statistics (SCB) data to:
    - Sample age from population distribution
    - Sample salary conditional on age (log-normal)
    - Derive balance from salary + structural position + noise
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        demographics_csv: str,
        seed: int = 0,
        balance_params: Dict = None,
        city_params: Dict = None
    ):
        """
        Initialize the demographics assigner.

        Args:
            graph: Transaction graph (after normal models built)
            demographics_csv: Path to demographics CSV file (required)
                Expected columns: age, average year income (tkr), median year income (tkr), population size
            seed: Random seed for reproducibility
            balance_params: Parameters for balance derivation:
                - balance_months: Target median balance as months of salary (default: 2.5)
                - alpha_salary: Salary coefficient (default: 0.6)
                - alpha_struct: Structural coefficient (default: 0.4)
                - sigma: Noise std dev (default: 0.5)
            city_params: Parameters for city assignment:
                - n_cities: Number of distinct cities (default: 10)
                - city_names: Optional list of city names (default: city_0, city_1, ...)
        """
        self.g = graph
        self.rng = np.random.default_rng(seed)

        # Load demographics data
        if not demographics_csv:
            raise ValueError("demographics_csv path is required")
        self.demographics = self._load_demographics(demographics_csv)

        # Balance derivation parameters
        # log(balance) = alpha_salary * log(salary) + alpha_struct * z(struct) + noise
        # Then shift so median balance = balance_months * median_salary
        self.balance_params = balance_params or {
            'balance_months': 2.5,  # Target median balance as months of median salary
            'alpha_salary': 0.6,    # Salary elasticity
            'alpha_struct': 0.4,    # Structural activity effect
            'sigma': 0.5            # Noise std dev
        }

        # City assignment parameters
        self.city_params = city_params or {
            'n_cities': 10,
            'city_names': None  # Will generate city_0, city_1, ... if None
        }

        # Compute population-weighted median salary for calibration
        self.median_monthly_salary = self._compute_population_median_salary()

        logger.info(f"Initialized DemographicsAssigner with {len(self.demographics)} age groups")

    def _load_demographics(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess demographics CSV."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Detect format: age bands (age_low, age_high) vs single ages (age)
        if 'age_low' in df.columns and 'age_high' in df.columns:
            self.age_band_format = True
            logger.info("Detected age band format (age_low, age_high)")
        elif 'age' in df.columns:
            self.age_band_format = False
            logger.info("Detected single-year age format")
        else:
            raise ValueError(
                "Demographics CSV must have 'age' column or 'age_low'/'age_high' columns. "
                "See module docstring for schema details."
            )

        # Validate required columns
        required = ['population size', 'average year income (tkr)', 'median year income (tkr)']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Demographics CSV missing required columns: {missing}")

        # Compute cumulative probabilities for age sampling
        total_pop = df['population size'].sum()
        df['cumulative_prob'] = df['population size'].cumsum() / total_pop

        return df

    def _compute_population_median_salary(self) -> float:
        """
        Compute the population-weighted median monthly salary from demographics.

        Uses median income per age bracket, weighted by population size.
        """
        df = self.demographics

        # Expand to population-weighted list of median incomes
        # (approximation: treat each age bracket's median as representative)
        total_pop = df['population size'].sum()
        cumulative_pop = 0
        median_yearly_tkr = None

        for _, row in df.iterrows():
            cumulative_pop += row['population size']
            if cumulative_pop >= total_pop / 2:
                median_yearly_tkr = row['median year income (tkr)']
                break

        if median_yearly_tkr is None or median_yearly_tkr <= 0:
            median_yearly_tkr = df['median year income (tkr)'].median()

        # Convert: yearly (tkr) -> monthly (SEK)
        median_monthly_sek = median_yearly_tkr * 1000 / 12

        logger.info(f"Population median monthly salary: {median_monthly_sek:.0f} SEK")
        return median_monthly_sek

    def assign(self, structural_scores: Dict = None):
        """
        Assign KYC attributes to all nodes in the graph.

        Args:
            structural_scores: Optional pre-computed {node: z_score} for structural position.
                              If None, computes log-degree z-scores.

                              For money laundering scenarios, consider passing a mixed score
                              combining multiple centrality measures (all z-scored):
                                - log-degree: local connectivity
                                - PageRank: global importance
                                - betweenness: broker potential

                              Example: weighted_z = 0.4*degree_z + 0.3*pagerank_z + 0.3*betweenness_z
        """
        logger.info("Assigning demographics-based KYC to nodes...")

        nodes = list(self.g.nodes())
        n_nodes = len(nodes)

        # Compute structural z-scores if not provided
        if structural_scores is None:
            structural_scores = self._compute_structural_zscores(nodes)

        # Sample age and salary for each node
        ages = []
        salaries = []

        for node in nodes:
            age, salary = self._sample_age_salary()
            ages.append(age)
            salaries.append(salary)

        # Compute balance from salary + structure + noise
        balances = self._compute_balances(
            salaries,
            [structural_scores.get(n, 0.0) for n in nodes]
        )

        # Assign age, salary, balance to nodes
        for i, node in enumerate(nodes):
            self.g.nodes[node]['age'] = ages[i]
            self.g.nodes[node]['salary'] = salaries[i]
            self.g.nodes[node]['init_balance'] = balances[i]

        # Assign cities with geographic clustering
        cities = self._assign_cities(nodes, structural_scores)
        for i, node in enumerate(nodes):
            self.g.nodes[node]['city'] = cities[i]

        # Log statistics
        self._log_statistics(ages, salaries, balances, cities, structural_scores, nodes)

        logger.info(f"Assigned KYC to {n_nodes} nodes")

    def _compute_structural_zscores(self, nodes: list) -> Dict:
        """Compute z-scored log-degree for structural position."""
        degrees = dict(self.g.degree())
        log_degrees = {n: np.log1p(degrees.get(n, 0)) for n in nodes}

        vals = np.array(list(log_degrees.values()))
        mu, sd = vals.mean(), vals.std() + 1e-12

        return {n: (log_degrees[n] - mu) / sd for n in nodes}

    def _sample_age_salary(self) -> tuple:
        """
        Sample age from population distribution, then salary conditional on age.

        Returns:
            (age, monthly_salary_sek)
        """
        # Sample age band based on population weights
        rand_val = self.rng.random()
        age_idx = np.searchsorted(self.demographics['cumulative_prob'].values, rand_val)
        age_idx = min(age_idx, len(self.demographics) - 1)  # Bounds check

        row = self.demographics.iloc[age_idx]

        # Get age: either single value or sample uniformly from band
        if self.age_band_format:
            age_low = int(row['age_low'])
            age_high = int(row['age_high'])
            age = self.rng.integers(age_low, age_high + 1)  # inclusive
        else:
            age = int(row['age'])

        # Get mean and median yearly income for this age
        mean_income = row['average year income (tkr)']
        median_income = row['median year income (tkr)']

        # Handle edge case where median is 0 (young ages)
        if median_income <= 0:
            median_income = max(mean_income * 0.8, 1.0)

        # Sample from log-normal distribution
        # For X ~ LogNormal(μ, σ²): median = exp(μ), mean = exp(μ + σ²/2)
        # So: σ² = 2(log(mean) - μ), but use max(mean, median) to handle noisy data
        mu = np.log(max(median_income, 1.0))
        sigma2 = max(0.0, 2 * (np.log(max(mean_income, median_income, 1.0)) - mu))
        sigma = np.sqrt(max(sigma2, 0.01))

        yearly_salary_tkr = self.rng.lognormal(mu, sigma)

        # Convert: yearly (tkr) -> monthly (SEK)
        monthly_salary_sek = yearly_salary_tkr * 1000 / 12

        return age, monthly_salary_sek

    def _compute_balances(self, salaries: list, struct_zscores: list) -> list:
        """
        Derive realistic balances from salary and structural position.

        log(balance) = alpha_salary * log(salary) + alpha_struct * z(struct) + noise
        Then shift so median balance = balance_months * population_median_salary.
        """
        params = self.balance_params
        n = len(salaries)

        # Noise term
        noise = self.rng.normal(0, params['sigma'], size=n)

        # Compute unshifted log balances
        log_balances = (
            params['alpha_salary'] * np.log1p(salaries) +
            params['alpha_struct'] * np.array(struct_zscores) +
            noise
        )

        # Calibrate: shift so median matches target (balance_months * median_salary)
        balance_months = params.get('balance_months', 2.5)
        target_median = balance_months * self.median_monthly_salary
        current_median = np.median(np.exp(log_balances))
        shift = np.log(target_median) - np.log(current_median)
        log_balances += shift

        # Convert to actual balance, ensure positive
        balances = np.exp(log_balances)
        balances = np.maximum(balances, 100)  # Minimum balance

        return balances.tolist()

    def _assign_cities(self, nodes: list, structural_scores: Dict) -> list:
        """
        Assign cities to nodes with geographic clustering.

        Uses BFS propagation from high-degree seed nodes to create realistic
        geographic clustering where nearby nodes (in network terms) share cities.

        Returns:
            List of city names, one per node (in same order as nodes list)
        """
        n_cities = self.city_params.get('n_cities', 10)
        city_names = self.city_params.get('city_names')

        if city_names is None:
            city_names = [f"city_{i}" for i in range(n_cities)]
        else:
            n_cities = len(city_names)

        n_nodes = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        cities = [None] * n_nodes

        # Select seed nodes: highest structural scores (hubs)
        # Use more seeds for larger graphs
        n_seeds = min(n_cities * 3, n_nodes // 10, n_nodes)
        n_seeds = max(n_seeds, n_cities)  # At least one seed per city

        sorted_nodes = sorted(nodes, key=lambda n: structural_scores.get(n, 0), reverse=True)
        seed_nodes = sorted_nodes[:n_seeds]

        # Assign cities to seeds (round-robin to ensure all cities used)
        for i, seed in enumerate(seed_nodes):
            city = city_names[i % n_cities]
            cities[node_to_idx[seed]] = city

        # BFS propagation: unassigned nodes inherit city from nearest assigned neighbor
        # Use undirected view for propagation (city spreads both ways)
        undirected = self.g.to_undirected()

        # Process nodes in random order to avoid systematic bias
        unassigned = [n for n in nodes if cities[node_to_idx[n]] is None]
        self.rng.shuffle(unassigned)

        # Multiple passes until all assigned (handles disconnected components)
        max_passes = 10
        for _ in range(max_passes):
            still_unassigned = []
            for node in unassigned:
                idx = node_to_idx[node]
                if cities[idx] is not None:
                    continue

                # Find neighbors with assigned cities
                neighbors = list(undirected.neighbors(node))
                neighbor_cities = [cities[node_to_idx[nb]] for nb in neighbors
                                   if nb in node_to_idx and cities[node_to_idx[nb]] is not None]

                if neighbor_cities:
                    # Take most common neighbor city (with random tiebreaker)
                    city_counts = {}
                    for c in neighbor_cities:
                        city_counts[c] = city_counts.get(c, 0) + 1
                    max_count = max(city_counts.values())
                    top_cities = [c for c, cnt in city_counts.items() if cnt == max_count]
                    cities[idx] = self.rng.choice(top_cities)
                else:
                    still_unassigned.append(node)

            if not still_unassigned:
                break
            unassigned = still_unassigned

        # Assign random city to any remaining unassigned (disconnected nodes)
        for node in unassigned:
            idx = node_to_idx[node]
            if cities[idx] is None:
                cities[idx] = self.rng.choice(city_names)

        logger.info(f"Assigned {n_cities} cities to {n_nodes} nodes")

        return cities

    def _log_statistics(self, ages, salaries, balances, cities, struct_scores, nodes):
        """Log sanity check statistics."""
        ages = np.array(ages)
        salaries = np.array(salaries)
        balances = np.array(balances)
        struct_vals = np.array([struct_scores.get(n, 0) for n in nodes])

        logger.info("=" * 60)
        logger.info("Demographics Assignment Statistics")
        logger.info("=" * 60)

        logger.info(f"Age: mean={ages.mean():.1f}, std={ages.std():.1f}, "
                   f"range=[{ages.min()}, {ages.max()}]")

        logger.info(f"Salary (monthly SEK): mean={salaries.mean():.0f}, "
                   f"median={np.median(salaries):.0f}, std={salaries.std():.0f}")

        logger.info(f"Balance: mean={balances.mean():.0f}, "
                   f"median={np.median(balances):.0f}, std={balances.std():.0f}")

        # Correlations (sanity checks)
        corr_age_balance = np.corrcoef(ages, np.log1p(balances))[0, 1]
        corr_salary_balance = np.corrcoef(np.log1p(salaries), np.log1p(balances))[0, 1]
        corr_struct_balance = np.corrcoef(struct_vals, np.log1p(balances))[0, 1]

        logger.info(f"Correlations with log(balance):")
        logger.info(f"  age: {corr_age_balance:.3f}")
        logger.info(f"  log(salary): {corr_salary_balance:.3f}")
        logger.info(f"  structural_z: {corr_struct_balance:.3f}")

        # City distribution
        from collections import Counter
        city_counts = Counter(cities)
        n_cities = len(city_counts)
        sizes = list(city_counts.values())
        logger.info(f"Cities: {n_cities} unique, sizes min={min(sizes)}, max={max(sizes)}, "
                   f"mean={np.mean(sizes):.0f}")

        logger.info("=" * 60)


def assign_kyc_from_demographics(
    graph: nx.DiGraph,
    demographics_csv: str,
    seed: int = 0,
    balance_params: Dict = None,
    city_params: Dict = None,
    structural_scores: Dict = None
):
    """
    Convenience function to assign demographics-based KYC to a graph.

    Call after normal models are built, before ML selector preparation.

    Args:
        graph: Transaction graph
        demographics_csv: Path to demographics CSV file (required).
            See module docstring for schema details.
        seed: Random seed
        balance_params: Balance derivation parameters (balance_months, alpha_salary, alpha_struct, sigma)
        city_params: City assignment parameters (n_cities, city_names)
        structural_scores: Pre-computed structural z-scores {node: z_score}.
            If None, uses log-degree z-scores. For AML scenarios, consider passing
            a mixed score (e.g., 0.4*degree_z + 0.3*pagerank_z + 0.3*betweenness_z).
    """
    assigner = DemographicsAssigner(
        graph=graph,
        demographics_csv=demographics_csv,
        seed=seed,
        balance_params=balance_params,
        city_params=city_params
    )
    assigner.assign(structural_scores=structural_scores)
