import numpy as np
import pandas as pd
import yaml

class BucketSimulation:
    def __init__(self, config, split):
        """
        Initializes the BucketSimulation class with split-specific configurations.
        """
        self.config = config['synthetic_data'][split]
        self.n_buckets = int(config['synthetic_data'][split]['n_buckets'])
        self.bucket_attributes_range = config['synthetic_data'][split]['bucket_attributes']
        self.rain_probability_range = config['synthetic_data'][split]['rain_probability']
        self.rain_depth_range = config['synthetic_data'][split]['rain_depth']
        self.time_step = float(config['time_step'])
        self.g = float(config['g'])
        self.is_noise = config['synthetic_data'][split].get('noise', False)
        self.buckets, self.h_water_level, self.mass_overflow = self.setup_buckets()
        self.noise_settings = config['synthetic_data'][split].get('noise', {})


    def setup_buckets(self):
        """
        Sets up initial conditions and attributes for buckets based on the range
        specified in the configuration.
        """
        buckets = {attr: np.random.uniform(float(low), float(high), self.n_buckets)
                   for attr, (low, high) in self.bucket_attributes_range.items()}
        h_water_level = np.array([np.random.uniform(0, value) for value in buckets["H_bucket"]])
        mass_overflow = np.random.random(self.n_buckets)
        return buckets, h_water_level, mass_overflow

    def pick_rain_params(self):
        """
        Randomly generates rain parameters based on configured probabilities and depths.
        """
        return [
            {key: [float(value[0]), float(value[1])] for key, value in self.rain_depth_range.items()},
            np.random.uniform(float(self.rain_probability_range["None"][0]), float(self.rain_probability_range["None"][1])),
            np.random.uniform(float(self.rain_probability_range["Heavy"][0]), float(self.rain_probability_range["Heavy"][1])),
            np.random.uniform(float(self.rain_probability_range["Light"][0]), float(self.rain_probability_range["Light"][1]))
        ]

    def simulate_rain_event(self, preceding_rain, rain_params):
        depth_range, no_rain_probability, heavy_rain_probability, light_rain_probability = rain_params
        if np.random.uniform(0.01, 0.99) < no_rain_probability:
            return 0
        elif preceding_rain < depth_range["Light"][1]:
            return np.random.uniform(*depth_range["Light"]) if np.random.uniform(0, 1) < light_rain_probability else np.random.uniform(*depth_range["Heavy"])
        else:
            return np.random.uniform(*depth_range["Heavy"]) if np.random.uniform(0, 1) < heavy_rain_probability else np.random.uniform(*depth_range["Light"])

    def generate_data(self, num_records):
        column_dtypes = {'precip': 'float64', 'et': 'float64', 'h_bucket': 'float64', 
                        'q_overflow': 'float64', 'q_spigot': 'float64'}
        data = pd.DataFrame(index=np.arange(num_records * self.n_buckets),
                            columns=column_dtypes.keys()).astype(column_dtypes)
        data['bucket_id'] = np.repeat(np.arange(self.n_buckets), num_records)
        data['time'] = np.tile(np.arange(num_records), self.n_buckets)

        # Initialize columns to zero using loc to avoid SettingWithCopyWarning
        data.loc[:, ['precip', 'et', 'h_bucket', 'q_overflow', 'q_spigot']] = 0

        for ibuc in range(self.n_buckets):
            for t in range(num_records):
                precip_in, et = self.simulate_rain_and_et(ibuc, t)
                self.process_respose_dynamics(ibuc, precip_in, et, t)
                spigot_out = self.calculate_spigot_out(ibuc, t)

                # Assign calculated values to the DataFrame
                idx = t + ibuc * num_records
                data.loc[idx, ['precip', 'et', 'h_bucket', 'q_overflow', 'q_spigot']] = [
                    precip_in, et, self.h_water_level[ibuc], self.mass_overflow[ibuc], spigot_out
                ]

                # Additional attributes can be filled in similarly if needed
                for attribute in self.bucket_attributes_range.keys():
                    data.loc[idx, attribute] = self.buckets[attribute][ibuc]

        return data


    def simulate_rain_and_et(self, ibuc, t):
        # Picking parameters for rain simulation
        rain_params = self.pick_rain_params()
        # Simulating the rain event
        if t == 0:
            preceding_rain = 0
        else:
            preceding_rain = self.h_water_level[ibuc]  # assuming rain affects the water level directly

        precip_in = self.simulate_rain_event(preceding_rain, rain_params)
        
        # Simulating evapotranspiration (ET)
        # ET (m/s) is the value at each time step taking diurnal fluctuations into account. The definite integral of the following function
        # (excluding noise) from 0 to 86400 is equal to ET_parameter, which is measured in m/day.
        et = max(0, (self.buckets["ET_parameter"][ibuc] * np.sin((1/13750.9870831) * t) *
                     np.random.normal(1, self.noise_settings.get('pet', 0))))
        
        return precip_in, et


    def process_respose_dynamics(self, ibuc, precip_in, et, t):
        # Updating water level with precipitation
        self.h_water_level[ibuc] += precip_in
        
        # Accounting for evaporation and infiltration

        # Calculate change in height due to infiltration using Darcy's Law. Divide Q (m^3/s) by A_bucket (m^2) to get infiltration (m/s)
        # Q = (k * rho * g * A_bucket * delta_h) / (mu * L)
        # infiltration = (k * rho * g * delta_h) / (mu * L)

        # k = K_infiltration = geologic permeability (m^2)
        # rho = density of water, constant, ~1000 kg/m^3
        # g = gravitational constant, ~9.807 m/s^2
        # mu = viscosity of water, constant , ~0.001 Pa/s
        # L = soil depth (m)
        # delta_h = hydraulic head/soil water potential (m) = soil depth + h_water_level
        # infitration = flow (m/s)

        k = self.buckets['K_infiltration'][ibuc]
        L = self.buckets['soil_depth'][ibuc]
        delta_h = self.h_water_level[ibuc] + L

        infiltration = (k * self.rho * self.g * delta_h) / (self.mu * L)

        self.h_water_level[ibuc] = np.max([0 , (self.h_water_level[ibuc] - et)])
        self.h_water_level[ibuc] = np.max([0 , (self.h_water_level[ibuc] - infiltration)])

        if self.is_noise:
            self.h_water_level[ibuc] *= np.random.normal(1, self.noise_settings.get('et', 0))
        
        # Checking for overflow
        if self.h_water_level[ibuc] > self.buckets['H_bucket'][ibuc]:
            self.mass_overflow[ibuc] = (self.h_water_level[ibuc] - self.buckets['H_bucket'][ibuc]) * self.buckets["A_bucket"][ibuc]
            self.h_water_level[ibuc] = self.buckets['H_bucket'][ibuc]
            if self.is_noise:
                self.h_water_level[ibuc] -= np.random.normal(0, self.noise_settings.get('q',0))
        else:
            self.mass_overflow[ibuc] = 0



    def calculate_spigot_out(self, ibuc, t):
        h_head_over_spigot = max(0, self.h_water_level[ibuc] - self.buckets['H_spigot'][ibuc])

        if self.is_noise:
            if h_head_over_spigot > 0:
                h_head_over_spigot = h_head_over_spigot * np.random.normal(1, self.noise_settings.get('head', 0))
            #elif h_head_over_spigot == 0:
            #    h_head_over_spigot = h_head_over_spigot + np.random.uniform(0, self.noise_settings.get('head', 0)/4)

        if h_head_over_spigot > 0:
            velocity_out = np.sqrt(2 * self.g * h_head_over_spigot)
            spigot_out = velocity_out * self.buckets['A_spigot'][ibuc] * self.time_step
            if self.is_noise:
                spigot_out = spigot_out * np.random.normal(1, self.noise_settings.get('q', 0))
            self.h_water_level[ibuc] = max(self.buckets["H_spigot"][ibuc], self.h_water_level[ibuc] - (spigot_out / self.buckets["A_bucket"][ibuc]))
        else:
            spigot_out = 0
            #if self.is_noise:
            #        spigot_out = spigot_out + np.random.uniform(1, self.noise_settings.get('q', 0)/4)

        return spigot_out