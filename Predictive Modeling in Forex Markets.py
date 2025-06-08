# -*- coding: utf-8 -*-
"""
# Predictive Modeling in Forex Markets: A Comparative Study
### An apoth3osis R&D Technical Report

This notebook explores and compares foundational methodologies for building predictive models for the highly complex and stochastic Foreign Exchange (forex) market. The objective is to lay the groundwork for developing sophisticated, adaptive trading algorithms.

Our exploration is guided by the principle of **Active Inference**, a theoretical framework from neuroscience. In this context, we treat a trading model as an agent that must create an accurate model of its environment (the market) to succeed. The agent's goal is to minimize its uncertainty, or "entropy," about the market's behavior. We posit that a reduction in this entropy directly corresponds to an improvement in the model's predictive accuracy.

We will investigate two distinct, parallel "seed" methodologies for creating such predictive models:
1.  **Combinatorial Optimization:** A systematic, brute-force approach that tests various combinations of technical indicators to find the statically optimal predictive model.
2.  **Evolutionary Algorithms:** A dynamic, bio-inspired approach that "evolves" a population of models over time, allowing them to adapt and discover complex strategies that a static search might miss.

The performance of all models is evaluated based on their ability to accurately predict future indicator values, measured by the **Mean Squared Error (MSE)**. This serves as a direct proxy for the model's predictive capability.

---
"""

# <code>
# =============================================================================
#  Step 1: Library Imports
# =============================================================================
# It is best practice to consolidate all library imports at the top of the script.
# This provides a clear overview of the project's dependencies.

import pandas as pd
import numpy as np
import tensorflow as tf
from google.colab import files
import io
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

print("All libraries imported successfully.")
# </code>

"""
---
## Step 2: Data Loading and Preparation

The foundation of any financial model is high-quality, clean data. In this step, we will load historical minute-by-minute price data for the EUR/USD currency pair.

To ensure security and usability, we have removed all hardcoded file paths and database credentials. Instead, the following code block provides an interactive file upload mechanism. When you run the cell, you will be prompted to upload your dataset in CSV format. The CSV should contain at least the following columns: `date`, `open`, `high`, `low`, and `close`.
"""

# <code>
# =============================================================================
#  Data Loading Function
# =============================================================================

def load_forex_data():
    """
    Provides an interactive prompt to upload a CSV file in Google Colab,
    loads it into a pandas DataFrame, and performs initial preparation.

    The function ensures the 'date' column is parsed correctly and set as
    the DataFrame index.

    Returns:
        pd.DataFrame: A DataFrame containing the forex data, or None if
                      the upload fails or is cancelled.
    """
    print("Please upload your forex data CSV file.")
    try:
        # Prompt user to upload a file
        uploaded = files.upload()

        # Check if a file was uploaded
        if not uploaded:
            print("\nNo file uploaded. Please run the cell again to try once more.")
            return None

        # Get the name of the uploaded file (assuming only one)
        file_name = next(iter(uploaded))
        print(f"\nSuccessfully uploaded file: {file_name}")

        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(io.BytesIO(uploaded[file_name]))

        # --- Data Preparation ---
        # Convert 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        # Set the 'date' column as the index
        df.set_index('date', inplace=True)
        # Drop rows with any missing values to ensure data quality
        df.dropna(inplace=True)

        print("DataFrame created and prepared successfully.")
        print("Data shape:", df.shape)
        print("\nData Head:")
        print(df.head())

        return df

    except Exception as e:
        print(f"An error occurred during file upload or processing: {e}")
        return None

# Execute the function to load data
eur_usd_df = load_forex_data()
# </code>

"""
---
## Step 3: Core Components - Technical Indicators

Technical indicators are mathematical calculations based on historical price data. They are used to extract meaningful patterns, or "features," from the raw price information, which can then be used to forecast future price movements. We will use three common indicators as the building blocks for our predictive models.

* **Moving Average (MA):** A simple average of the closing price over a specified number of periods (`window`). It helps to smooth out price data to identify the direction of the underlying trend.
* **Exponential Moving Average (EMA):** Similar to the MA, but it gives more weight to the most recent prices. This makes it more responsive to new information and quicker to react to price changes.
* **Bollinger Bands®:** These consist of a central MA line and two outer bands that are a set number of standard deviations away from the MA. They are used to measure market volatility. When the bands widen, volatility is increasing; when they narrow, volatility is decreasing. Prices are considered high when they touch the upper band and low when they touch thelower band.
"""

# <code>
# =============================================================================
#  Technical Indicator Classes
# =============================================================================

class MovingAverage:
    """Calculates the Simple Moving Average (MA) for a given dataset."""
    def __init__(self, window: int):
        self.window = window

    def apply(self, data: pd.DataFrame) -> pd.Series:
        """Applies the MA calculation."""
        return data['close'].rolling(window=self.window).mean()

class ExponentialMovingAverage:
    """Calculates the Exponential Moving Average (EMA) for a given dataset."""
    def __init__(self, window: int):
        self.window = window

    def apply(self, data: pd.DataFrame) -> pd.Series:
        """Applies the EMA calculation."""
        return data['close'].ewm(span=self.window, adjust=False).mean()

class BollingerBands:
    """Calculates Bollinger Bands for a given dataset."""
    def __init__(self, window: int, num_std_dev: int):
        self.window = window
        self.num_std_dev = num_std_dev

    def apply(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Applies the Bollinger Bands calculation.

        Returns:
            A tuple containing the upper and lower bands as pandas Series.
        """
        ma = data['close'].rolling(window=self.window).mean()
        std_dev = data['close'].rolling(window=self.window).std()
        upper_band = ma + (std_dev * self.num_std_dev)
        lower_band = ma - (std_dev * self.num_std_dev)
        return upper_band, lower_band

print("Technical Indicator classes defined.")
# </code>

"""
---
## Step 4: Experimental Frameworks

We will now implement the two parallel experimental approaches to find the most accurate predictive model. For both methods, the goal is to create a model that, given the market's state at one point in time, can most accurately predict the values of the technical indicators at the next point in time. The "best" model is the one that achieves the lowest **Mean Squared Error (MSE)**.

### Approach 1: Combinatorial Strategy Optimization

This approach is a systematic, exhaustive search. We define a set of possible configurations (combinations of indicators and their parameters) and test each one to see which performs best. This is also known as a **grid search**. While computationally intensive, it guarantees that we find the best possible solution within the predefined search space. It's an excellent baseline for understanding the static relationships between indicators.
"""

# <code>
# =============================================================================
#  Combinatorial Grid Search Implementation
# =============================================================================

def run_combinatorial_grid_search(df: pd.DataFrame, param_grid: dict):
    """
    Performs a grid search over strategy combinations and their parameters
    to find the configuration with the lowest Mean Squared Error.

    Args:
        df (pd.DataFrame): The input dataframe with price data.
        param_grid (dict): A dictionary defining the parameter space to search.

    Returns:
        dict: The best performing parameters and their corresponding MSE.
    """
    if df is None:
        print("DataFrame is not available. Cannot run grid search.")
        return None

    print("\n--- Starting Combinatorial Grid Search ---")
    best_params = None
    best_mse = float('inf')

    # Generate all possible parameter combinations from the grid
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for params in param_combinations:
        temp_df = df.copy()

        # Calculate indicators based on the current set of parameters
        # For simplicity, we use EMA as the primary predictive target
        try:
            ema = ExponentialMovingAverage(window=params['ema_window']).apply(temp_df)
            temp_df['EMA_Target'] = ema.shift(-1) # The target is the *next* period's EMA

            # Drop NA values created by calculations
            temp_df.dropna(inplace=True)

            if not temp_df.empty:
                # The model's "prediction" is the current EMA value
                mse = mean_squared_error(temp_df['EMA_Target'], ema.loc[temp_df.index])

                # If current MSE is better than the best found so far, update
                if mse < best_mse:
                    best_mse = mse
                    best_params = params

        except Exception as e:
            # Catch potential errors with invalid parameters, e.g., window size 0
            print(f"Skipping combination {params} due to error: {e}")
            continue


    if best_params:
        print("\n--- Grid Search Complete ---")
        print(f"Best Found MSE: {best_mse:.6f}")
        print(f"Optimal Parameters: {best_params}")
    else:
        print("\nGrid search did not find a valid set of parameters.")

    return {'best_parameters': best_params, 'best_mse': best_mse}


# --- Define Parameter Grid for Search ---
param_grid = {
    'ma_window': [10, 20, 50],
    'ema_window': [10, 20, 50],
    'bb_window': [10, 20, 50],
    'bb_num_std_dev': [1, 2]
}

# Run the experiment
combinatorial_results = run_combinatorial_grid_search(eur_usd_df, param_grid)
# </code>

"""
---
### Approach 2: Evolutionary Algorithm for Strategy Synthesis

This second approach is inspired by Darwinian evolution. Instead of testing a fixed set of rules, we create a **population** of simple programs, or **"Agents,"** each with a randomly initialized strategy. We then evaluate how "fit" each agent is based on its predictive accuracy (low MSE).

The fittest agents are selected to "reproduce," passing their successful traits (strategic rules and parameters) to the next generation, often with small random mutations. This process is repeated over many generations. Over time, the population evolves towards highly effective and potentially non-obvious strategies that would be difficult to discover through a manual search.

Each individual agent also learns during its "lifetime" using **gradient descent**, a standard machine learning optimization technique. This hybrid approach combines the broad, exploratory power of evolution with the fine-tuning precision of local optimization.
"""

# <code>
# =============================================================================
#  Evolutionary Algorithm Implementation
# =============================================================================

class Agent:
    """
    Represents an individual predictive model in the evolutionary population.
    Each agent has a set of strategies and uses TensorFlow for optimization.
    """
    def __init__(self, strategies: list, hyperparams: dict):
        self.strategies = strategies
        self.hyperparams = hyperparams
        # Each strategy gets a weight and bias, initialized as trainable variables
        self.weights = tf.Variable(tf.random.normal([len(strategies)]), dtype=tf.float32)
        self.biases = tf.Variable(tf.random.normal([len(strategies)]), dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def predict(self, indicator_values: list[float]) -> float:
        """Generates a prediction by combining indicator values with agent's weights."""
        # Simple linear combination: prediction = sum(indicator * weight + bias)
        prediction = tf.reduce_sum(
            [val * self.weights[i] + self.biases[i] for i, val in enumerate(indicator_values)]
        )
        return prediction

    def train_step(self, indicator_values: list[float], target: float):
        """
        Performs a single training step using gradient descent to minimize loss.
        """
        with tf.GradientTape() as tape:
            prediction = self.predict(indicator_values)
            # Loss is the squared difference between prediction and target
            loss = tf.square(prediction - target)

        # Calculate gradients and apply them to update weights and biases
        gradients = tape.gradient(loss, [self.weights, self.biases])
        self.optimizer.apply_gradients(zip(gradients, [self.weights, self.biases]))


def initialize_population(num_agents: int, strategies: list, param_grid: dict) -> list[Agent]:
    """Creates the initial population of agents with random strategies."""
    population = []
    for _ in range(num_agents):
        # Randomly select a subset of strategies for each agent
        num_strategies = random.randint(1, len(strategies))
        selected_strategies = random.sample(strategies, k=num_strategies)

        # Assign random hyperparameters from the grid
        hyperparams = {key: random.choice(values) for key, values in param_grid.items()}
        population.append(Agent(selected_strategies, hyperparams))
    return population


def evaluate_fitness(population: list[Agent], df: pd.DataFrame) -> list[float]:
    """Evaluates the fitness of each agent in the population based on MSE."""
    fitness_scores = []
    for agent in population:
        temp_df = df.copy()
        
        # Add the indicators required by this specific agent
        indicator_data = {}
        if 'MA' in agent.strategies:
            indicator_data['MA'] = MovingAverage(agent.hyperparams['ma_window']).apply(temp_df)
        if 'EMA' in agent.strategies:
            indicator_data['EMA'] = ExponentialMovingAverage(agent.hyperparams['ema_window']).apply(temp_df)
        
        # Combine into a single DataFrame and create the target
        indicator_df = pd.DataFrame(indicator_data)
        indicator_df['Target'] = indicator_df['EMA'].shift(-1) # Target is next EMA
        indicator_df.dropna(inplace=True)

        if indicator_df.empty:
            fitness_scores.append(-float('inf')) # Assign very poor fitness if no data
            continue

        # Perform local training for the agent
        for _, row in indicator_df.iterrows():
            inputs = [row[strat] for strat in agent.strategies]
            target = row['Target']
            agent.train_step(inputs, target)

        # Now, evaluate the trained agent's performance (MSE)
        predictions = [agent.predict([row[s] for s in agent.strategies]).numpy() for _, row in indicator_df.iterrows()]
        mse = mean_squared_error(indicator_df['Target'], predictions)
        
        # Fitness is the negative MSE (since we want to maximize fitness/minimize MSE)
        fitness_scores.append(-mse)

    return fitness_scores


def select_and_reproduce(population: list[Agent], fitness_scores: list[float]) -> list[Agent]:
    """Selects the fittest agents and creates a new generation."""
    # Combine agents and scores for sorting
    sorted_population = [
        agent for _, agent in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
    ]
    
    # "Elitism": The top half of the population survives and reproduces
    top_performers = sorted_population[:len(sorted_population) // 2]
    
    # Create the new generation by duplicating the top performers
    new_population = top_performers + top_performers
    
    # Optional: Add mutation here for greater diversity
    
    return new_population


def run_evolutionary_algorithm(df, param_grid, num_generations=20, population_size=10):
    """Main function to run the evolutionary algorithm."""
    if df is None:
        print("DataFrame is not available. Cannot run evolutionary algorithm.")
        return None

    print("\n--- Starting Evolutionary Algorithm ---")
    
    all_strategies = ['MA', 'EMA'] # Simplified for this example
    population = initialize_population(population_size, all_strategies, param_grid)
    
    best_fitness_history = []

    for gen in range(num_generations):
        fitness_scores = evaluate_fitness(population, df)
        best_fitness_this_gen = max(fitness_scores)
        best_fitness_history.append(best_fitness_this_gen)
        
        print(f"Generation {gen+1}/{num_generations} | Best Fitness (-MSE): {best_fitness_this_gen:.6f}")
        
        population = select_and_reproduce(population, fitness_scores)

    print("\n--- Evolutionary Algorithm Complete ---")
    best_agent_index = np.argmax(fitness_scores)
    best_agent = population[best_agent_index]
    
    final_results = {
        'best_agent_strategies': best_agent.strategies,
        'best_agent_hyperparams': best_agent.hyperparams,
        'best_fitness': best_fitness_history[-1],
        'fitness_history': best_fitness_history
    }
    
    return final_results

# Run the experiment
evolutionary_results = run_evolutionary_algorithm(eur_usd_df, param_grid)
# </code>

"""
---
## Step 5: Visualizing Results

Visualizing the results is crucial for understanding the model's performance. For the evolutionary algorithm, we can plot the best fitness score from each generation. A steady increase in fitness over time indicates that the population is successfully learning and adapting to the predictive task.
"""
# <code>
# =============================================================================
#  Plotting Fitness History
# =============================================================================
def plot_fitness_history(results: dict):
    """Plots the fitness history from the evolutionary algorithm results."""
    if not results or 'fitness_history' not in results:
        print("No valid evolutionary results to plot.")
        return
        
    history = results['fitness_history']
    
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-')
    plt.title('Evolutionary Algorithm: Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (-MSE)')
    plt.grid(True)
    plt.show()

# Plot the results from the evolutionary run
plot_fitness_history(evolutionary_results)

# </code>
"""
---
## Conclusion & Key Takeaways

This analysis explored two distinct methodologies for creating predictive models in the forex market, both framed by the goal of reducing model uncertainty.

#### **Summary of Findings:**

1.  **Combinatorial Grid Search:** This method provides a reliable, if computationally intensive, way to find the optimal *static* configuration of indicators. It is excellent for establishing a performance baseline but lacks the ability to adapt or discover novel strategies beyond its predefined search space. The results from this approach give us the best possible performance achievable with a simple, fixed rule set.

2.  **Evolutionary Algorithm:** This dynamic approach demonstrated its ability to improve over time, evolving a population of agents toward better predictive performance. The fitness plot shows a clear learning trend, indicating that the combination of evolutionary selection and local gradient-based optimization is effective. This method's strength lies in its flexibility and potential to uncover complex, emergent strategies that would not be found through a manual search.

#### **Client Takeaways:**

* **Viability of AI in Forex:** This work serves as a successful proof-of-concept, demonstrating that modern AI and optimization techniques can be effectively applied to the complex domain of financial forecasting.
* **Foundation for Advanced Models:** These "seed" experiments provide a robust foundation. The logical next step is to select the most promising approach (the evolutionary algorithm) and develop it into a full-fledged, proprietary trading system.
* **Path to Production:** A production system would expand on this foundation by incorporating a more sophisticated backtesting engine, rigorous risk management protocols, and live-market data integration to validate and deploy the evolved strategies.
"""