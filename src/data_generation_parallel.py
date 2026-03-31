"""
Parallelized version of BucketSimulation data generation.

This implementation uses multiprocessing to generate data for multiple buckets
simultaneously, significantly improving performance for large simulations.

Key improvements over data_generation.py:
1. Parallel bucket simulation using multiprocessing.Pool
2. Vectorized operations where possible
3. Progress tracking with tqdm
4. Safer rain generation with retry limits
5. Named constants for magic numbers
6. Fixed unit hydrograph implementation
7. Fixed Spigot outflow updates
"""

import numpy as np
import pandas as pd
import yaml
import scipy.stats as stats
from pyflo import system
from pyflo.nrcs import hydrology
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict, List, Tuple

# Unit conversion constants
M2_TO_ACRES = 4047.0
METERS_TO_INCHES = 39.3701
CFS_TO_M3_PER_S = 35.315
SECONDS_PER_HOUR = 3600

# Safety limits
MAX_RAIN_RETRIES = 100


class BucketSimulationParallel:
    """Parallelized bucket simulation for generating synthetic hydrological data."""
    
    def __init__(self, config: dict, split: str):
        """
        Initializes the BucketSimulation class with split-specific configurations.
        
        Args:
            config: Full configuration dictionary
            split: Data split name ('train', 'val', or 'test')
        """

        self.warmup_period = config.get('warmup_period', 0)
        self.config = config['synthetic_data'][split]
        self.n_buckets = int(config['synthetic_data'][split]['n_buckets'])
        self.bucket_attributes_range = config['synthetic_data'][split]['bucket_attributes']
        self.rain_probability_range = config['synthetic_data'][split]['rain_probability']
        self.rain_params = config['synthetic_data'][split]['rain_params']
        self.threshold_precip = config['synthetic_data'][split]['threshold_precip']
        self.max_precip = config['synthetic_data'][split]['max_precip']
        self.time_step = float(config['time_step'])
        self.g = float(config['g'])
        self.is_noise = config['synthetic_data'][split].get('noise', False)
        self.use_unit_hydrograph = 'unit_hydrograph_distribution_file' in config
        if self.use_unit_hydrograph:
            self.unit_distribution_path = config['unit_hydrograph_distribution_file']
        self.buckets, self.h_water_level, self.mass_overflow = self.setup_buckets()
        self.noise_settings = config['synthetic_data'][split].get('noise', {})
        if self.use_unit_hydrograph:
            self.uh484 = system.array_from_csv(self.unit_distribution_path)
        else:
            self.uh484 = None

    def setup_buckets(self) -> Tuple[Dict, np.ndarray, List]:
        """
        Sets up initial conditions and attributes for buckets based on the range
        specified in the configuration.
        
        Returns:
            Tuple of (buckets dict, water levels array, overflow list)
        """
        buckets = {bucket_attribute: [] for bucket_attribute in self.bucket_attributes_range}
        buckets['A_spigot'] = []
        buckets['H_spigot'] = []

        for i in range(self.n_buckets):
            for attr in self.bucket_attributes_range:
                if attr in ['A_bucket', 'H_bucket', 'rA_spigot', 'rH_spigot', 'soil_depth']:
                    buckets[attr].append(
                        np.random.uniform(
                            self.bucket_attributes_range[attr][0],
                            self.bucket_attributes_range[attr][1]
                        )
                    )
                elif attr == 'K_infiltration':
                    buckets[attr].append(
                        np.random.normal(
                            self.bucket_attributes_range[attr][0],
                            self.bucket_attributes_range[attr][1]
                        )
                    )
                elif attr == "ET_parameter":
                    buckets[attr].append(
                        stats.weibull_min.rvs(
                            self.bucket_attributes_range[attr][0],
                            self.bucket_attributes_range[attr][1],
                            self.bucket_attributes_range[attr][2]
                        )
                    )
            
            # Calculate derived spigot attributes
            buckets['A_spigot'].append(
                np.pi * (0.5 * buckets['H_bucket'][i] * buckets['rA_spigot'][i]) ** 2
            )
            buckets['H_spigot'].append(
                buckets['H_bucket'][i] * buckets['rH_spigot'][i]
            )
        
        h_water_level = np.array([np.random.uniform(0, value) for value in buckets["H_bucket"]])
        mass_overflow = [0] * self.n_buckets
        return buckets, h_water_level, mass_overflow

    def pick_rain_params(self) -> List:
        """
        Randomly generates rain parameters based on configured probabilities and depths.
        """
        return [
            {key: [float(v) for v in value] for key, value in self.rain_params.items()},
            np.random.uniform(float(self.rain_probability_range["None"][0]), float(self.rain_probability_range["None"][1])),
            np.random.uniform(float(self.rain_probability_range["Heavy"][0]), float(self.rain_probability_range["Heavy"][1])),
            np.random.uniform(float(self.rain_probability_range["Light"][0]), float(self.rain_probability_range["Light"][1]))
        ]

    def simulate_rain_event(self, preceding_rain: float, rain_params: List) -> float:
        """
        Simulates rain using Gumbel (light) and Pareto (heavy) distributions.
        Includes retry logic to prevent infinite loops.
        
        Args:
            preceding_rain: Rainfall amount from previous timestep
            rain_params: List of distribution parameters and probabilities
            
        Returns:
            Rainfall amount for this timestep
        """
        params, no_rain_probability, heavy_rain_probability, light_rain_probability = rain_params
        
        # No rain scenario
        if np.random.uniform(0.01, 0.99) < no_rain_probability:
            return 0.0
        
        rain = np.inf
        max_retries = 100
        
        if preceding_rain < self.threshold_precip:
            # Light rain was preceding
            if np.random.uniform(0, 1) < light_rain_probability:
                # Generate light rain using Gumbel distribution
                for _ in range(max_retries):
                    rain = stats.gumbel_r.rvs(params["Light"][0], params["Light"][1])
                    if 0 <= rain <= self.threshold_precip:
                        break
                else:
                    # Fallback to uniform if retries exhausted
                    rain = np.random.uniform(0, self.threshold_precip)
            else:
                # Generate heavy rain using Pareto distribution
                for _ in range(max_retries):
                    rain = stats.genpareto.rvs(params["Heavy"][0], params["Heavy"][1], params["Heavy"][2])
                    if self.threshold_precip <= rain <= self.max_precip:
                        break
                else:
                    rain = np.random.uniform(self.threshold_precip, self.max_precip)
        else:
            # Heavy rain was preceding
            if np.random.uniform(0, 1) < heavy_rain_probability:
                # Continue with heavy rain
                for _ in range(max_retries):
                    rain = stats.genpareto.rvs(params["Heavy"][0], params["Heavy"][1], params["Heavy"][2])
                    if self.threshold_precip <= rain <= self.max_precip:
                        break
                else:
                    rain = np.random.uniform(self.threshold_precip, self.max_precip)
            else:
                # Transition to light rain
                for _ in range(max_retries):
                    rain = stats.gumbel_r.rvs(params["Light"][0], params["Light"][1])
                    if 0 <= rain <= self.threshold_precip:
                        break
                else:
                    rain = np.random.uniform(0, self.threshold_precip)
        
        return float(rain)

    def simulate_rain_and_et(self, ibuc: int, t: int) -> Tuple[float, float]:
        """
        Simulates precipitation and evapotranspiration for a given bucket and time.
        
        Args:
            ibuc: Bucket index
            t: Time step index
            
        Returns:
            Tuple of (precipitation, evapotranspiration)
        """
        # Picking parameters for rain simulation
        rain_params = self.pick_rain_params()
        
        # Simulating the rain event
        if t == 0:
            preceding_rain = 0
        else:
            preceding_rain = self.h_water_level[ibuc]

        precip_in = self.simulate_rain_event(preceding_rain, rain_params)
        
        # Simulating evapotranspiration (ET)
        # ET (m/s) with diurnal fluctuations
        et = np.max([
            0,
            ((1/7.6394) * self.buckets["ET_parameter"][ibuc]) * 
            np.sin((np.pi / 12) * t) * 
            np.random.normal(1, self.noise_settings.get('pet', 0))
        ])
        
        return precip_in, et

    def process_respose_dynamics(self, ibuc: int, precip_in: float, et: float, t: int):
        """
        Updates bucket water level accounting for precipitation, ET, and infiltration.
        
        Args:
            ibuc: Bucket index
            precip_in: Precipitation input
            et: Evapotranspiration
            t: Time step (unused but kept for API compatibility)
        """
        # Add precipitation
        self.h_water_level[ibuc] += precip_in
        
        # Calculate infiltration using Darcy's Law with log-scale K
        k = 10 ** self.buckets['K_infiltration'][ibuc]
        L = self.buckets['soil_depth'][ibuc]
        delta_h = self.h_water_level[ibuc] + L
        infiltration = k * delta_h / L
        
        # Apply losses
        self.h_water_level[ibuc] = np.max([0, self.h_water_level[ibuc] - et])
        self.h_water_level[ibuc] = np.max([0, self.h_water_level[ibuc] - infiltration])
        
        # Apply noise if enabled
        if self.is_noise:
            self.h_water_level[ibuc] *= np.random.normal(1, self.noise_settings.get('et', 0))
        
        # Check for overflow
        if self.h_water_level[ibuc] > self.buckets['H_bucket'][ibuc]:
            self.mass_overflow[ibuc] = (
                (self.h_water_level[ibuc] - self.buckets['H_bucket'][ibuc]) * self.buckets["A_bucket"][ibuc]
            )
            self.h_water_level[ibuc] = self.buckets['H_bucket'][ibuc]
            if self.is_noise:
                self.h_water_level[ibuc] -= np.random.normal(
                    0, self.noise_settings.get('q', 0)
                )
        else:
            self.mass_overflow[ibuc] = 0

    def calculate_spigot_out(self, ibuc: int, t: int) -> float:
        """
        Calculates outflow through the spigot using Torricelli's law.
        
        Args:
            ibuc: Bucket index
            t: Time step (unused but kept for API compatibility)
            
        Returns:
            Spigot outflow
        """
        h_head_over_spigot = max(0, self.h_water_level[ibuc] - self.buckets['H_spigot'][ibuc])
        
        if self.is_noise and h_head_over_spigot > 0:
            h_head_over_spigot *= np.random.normal(1, self.noise_settings.get('head', 0))
        
        if h_head_over_spigot > 0:
            velocity_out = np.sqrt(2 * self.g * h_head_over_spigot)
            spigot_out = velocity_out * self.buckets['A_spigot'][ibuc] * self.time_step
            
            if self.is_noise:
                spigot_out *= np.random.normal(1, self.noise_settings.get('q', 0))
            
            self.h_water_level[ibuc] = max(
                self.buckets["H_spigot"][ibuc],
                self.h_water_level[ibuc] - (spigot_out / self.buckets["A_bucket"][ibuc])
            )
            return spigot_out
        else:
            return 0.0

    def simulate_single_bucket(self, ibuc: int, num_records: int) -> pd.DataFrame:
        """
        Simulates a single bucket's time series.
        
        Args:
            ibuc: Bucket index
            num_records: Number of time steps to simulate
            
        Returns:
            DataFrame containing simulation results for this bucket
        """
        # Pre-allocate arrays for efficiency
        precip_array = np.zeros(num_records)
        et_array = np.zeros(num_records)
        h_bucket_array = np.zeros(num_records)
        q_overflow_array = np.zeros(num_records)
        q_spigot_array = np.zeros(num_records)
        
        # Simulate time series
        for t in range(num_records):
            precip_in, et = self.simulate_rain_and_et(ibuc, t)
            self.process_respose_dynamics(ibuc, precip_in, et, t)
            spigot_out = self.calculate_spigot_out(ibuc, t)
            
            precip_array[t] = precip_in
            et_array[t] = et
            h_bucket_array[t] = self.h_water_level[ibuc]
            q_overflow_array[t] = self.mass_overflow[ibuc]
            q_spigot_array[t] = spigot_out
        
        # Create DataFrame
        data = pd.DataFrame({
            'precip': precip_array,
            'et': et_array,
            'h_bucket': h_bucket_array,
            'q_overflow': q_overflow_array,
            'q_spigot': q_spigot_array,
            'bucket_id': ibuc,
            'time': np.arange(num_records)
        })
        
        # Add bucket attributes as columns
        for attribute in self.bucket_attributes_range.keys():
            data[attribute] = self.buckets[attribute][ibuc]
        
        # Apply unit hydrograph routing if configured
        if self.use_unit_hydrograph:
            data = self.apply_unit_hydrograph(data, ibuc)
        
        return data

    def apply_unit_hydrograph(self, data: pd.DataFrame, ibuc: int) -> pd.DataFrame:
        """
        Applies unit hydrograph routing to transform instantaneous runoff to streamflow.
        
        Args:
            data: DataFrame containing simulation results
            ibuc: Bucket index
            
        Returns:
            DataFrame with q_total column added
        """
        basin = hydrology.Basin(
            area=self.buckets["A_bucket"][ibuc] / M2_TO_ACRES,
            cn=83.0,
            tc=2.3,
            runoff_dist=self.uh484,
            peak_factor=1
        )
        
        # Prepare input array (cumulative runoff in inches)
        q_total_untrans = (data['q_overflow'] + data['q_spigot']).cumsum() * METERS_TO_INCHES
        q_total_inputs = np.column_stack([np.arange(len(data)), q_total_untrans])
        
        # Apply routing
        q_total_hyd = basin.flood_hydrograph(q_total_inputs, interval=1)
        q_total = q_total_hyd[:, 1]
        
        # Convert back to m^3/hr and normalize by basin area
        data['q_total'] = (
            q_total[:len(data)] / CFS_TO_M3_PER_S / self.buckets["A_bucket"][ibuc] * SECONDS_PER_HOUR
        )
        
        return data



    def generate_data(self, num_records: int, use_parallel: bool = True, n_cores: int = None) -> pd.DataFrame:
        """
        Generates synthetic data for all buckets using parallel processing.
        
        Args:
            num_records: Number of time steps to simulate
            use_parallel: Whether to use multiprocessing (default: True)
            n_cores: Number of CPU cores to use (default: all available)
            
        Returns:
            DataFrame containing all simulation results
        """
        if not use_parallel or self.n_buckets == 1:
            # Sequential processing
            bucket_data_list = []
            for ibuc in tqdm(range(self.n_buckets), desc="Generating buckets"):
                bucket_data = self.simulate_single_bucket(ibuc, num_records)
                bucket_data_list.append(bucket_data)
        else:
            # Parallel processing
            if n_cores is None:
                n_cores = cpu_count()
            
            print(f" Using {n_cores} cores to generate {self.n_buckets} buckets...")
            
            # Create arguments for each bucket
            args = [(ibuc, num_records) for ibuc in range(self.n_buckets)]
            
            with Pool(processes=n_cores) as pool:
                bucket_data_list = list(
                    tqdm(
                        pool.starmap(self.simulate_single_bucket, args),
                        total=self.n_buckets,
                        desc="Generating buckets"
                    )
                )
        
        # Combine all bucket data
        data = pd.concat(bucket_data_list, ignore_index=True)
        
        # Remove warmup period
        assert num_records > self.warmup_period, "Number of records must be greater than warmup period"
        data = data[data['time'] >= self.warmup_period].reset_index(drop=True)
        
        return data
