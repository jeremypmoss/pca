import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader
import quasar_functions as qf
from sklearn.inspection import permutation_importance
import seaborn as sns
from scipy import stats


# Set seeds for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Set up the style for plotting
rc_fonts = {
    "text.usetex": False,
    'font.family': 'serif',
    'font.size': 20,
}

# Merge the dictionaries
rc_params = {'figure.autolayout': True, **rc_fonts}

# Update the matplotlib rcParams
plt.rcParams.update(rc_params)

#%% Load data using DataLoader
class_params = {'dropna': False,
                'colours': False,
                'impute_method': 19.1} # None, 'max', 'mean' or float
load_params = {'name': 'sdssmags',
               'number_of_rows': None,
               'binning': [0.1, 0.2, 0.3, 0.4],
               'selected_bin': None}

dl = DataLoader(**class_params)
dataset, datasetname, magnames, mags = dl.load_data(**load_params)

# Split the data into training and testing sets
X = mags
y = dataset['redshift']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

apply_neural_network = True
check_colinearity = True
handle_outliers = False

#%% Looking specifically at turnover flag
# Splitting the data based on TO_flag
if 'TO_flag' in dataset.columns: # only mq_x_gleam_all has this feature so far
    TO_flag_1 = dataset[dataset['TO_flag'] == 1]
    TO_flag_0 = dataset[dataset['TO_flag'] == 0]

    # Summary statistics
    print("TO_flag=1 Summary Statistics:")
    print(TO_flag_1['redshift'].describe())

    print("\nTO_flag=0 Summary Statistics:")
    print(TO_flag_0['redshift'].describe())

    # Performing statistical test (e.g., t-test)
    t_stat, p_value = stats.ttest_ind(TO_flag_1['redshift'], TO_flag_0['redshift'], equal_var=False)
    print("\nT-test p-value:", p_value)

    # Boxplot

    colors = {0: 'red', 1: 'green'} # define a color palette for the boxplot

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='TO_flag', y='redshift', data=dataset, palette=colors)
    plt.grid(True)
    plt.xlabel('TO_flag')
    plt.ylabel('Redshift')
    plt.title('Boxplot for\nTurnover Frequency Presence with Redshift')
    plt.show()

    # Histograms
    plt.figure(figsize=(8, 6))
    plt.hist(TO_flag_0['redshift'], bins=50, alpha=1, label='Turnover frequency absent', color = colors[0])
    plt.hist(TO_flag_1['redshift'], bins=50, alpha=1, label='Turnover frequency present', color = colors[1])
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.title('Redshift Distribution for\nTurnover Frequency Presence with Redshift')
    plt.show()

    # Density plots
    plt.figure(figsize=(8, 6))

    TO_flag_1['redshift'].plot(kind='density', label='Turnover frequency present', color = colors[1])
    TO_flag_0['redshift'].plot(kind='density', label='Turnover frequency absent', color = colors[0])
    plt.xlabel('Redshift')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()
    plt.title('Redshift Density for\nTurnover Frequency Presence with Redshift')
    plt.show()


#%% Standardize all datasets and target
# Store column names before scaling
column_names = X_train.columns

# Standardize the data after preprocessing
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if check_colinearity:
    # The following few lines check for for multi-colinearity
    # Convert scaled arrays back to DataFrames with column names
    X_train = pd.DataFrame(X_train_scaled, columns=column_names)
    X_test = pd.DataFrame(X_test_scaled, columns=column_names)

    # # Check for highly correlated features and remove them
    correlation_matrix = X_train.corr().abs()
    upper_triangle = np.triu(correlation_matrix, k=1)
    to_drop = [column for column in X_train.columns if any(upper_triangle[X_train.columns.get_loc(column)] > 0.95)]
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    print(f'The following columns appear to be highly corelated and will be dropped:\n{to_drop}')

#%% Outlier handling

if handle_outliers:
# Identify and remove outliers
    z_scores = stats.zscore(y_train)
    outlier_indices = np.where(np.abs(z_scores) > 3)[0]
    X_train = np.delete(X_train, outlier_indices, axis=0)
    y_train = np.delete(y_train, outlier_indices)

    # In here, I need to check the state of y_pred
#%% Apply PCA to the necessary datasets

from sklearn.model_selection import GridSearchCV

# Create a PCA instance without specifying the number of components
pca = PCA()
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed_value)

# Define a range of percentages of explained variance to consider
# For example, you can consider 95%, 98%, 99%, or any other desired value
variance_thresholds = [0.95, 0.98, 0.99]

# Create a parameter grid for GridSearchCV to search through
param_grid = {'n_components': [None] + [int(X_train.shape[1] * threshold) for threshold in variance_thresholds]}

# Perform a grid search with cross-validation to find the best number of components
grid_search = GridSearchCV(estimator=pca, param_grid=param_grid, cv=cv)
grid_search.fit(X_train_scaled)

# Get the best number of components
best_n_components = grid_search.best_params_['n_components']
# best_n_components = 2

# Apply PCA with the best number of components to the data
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check explained variance
print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')

# Now, best_n_components contains the optimal number of components
print(f'Optimal number of components: {best_n_components}')


#%% Compare regression models

# Define ranges of hyperparameters for the models
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed_value)
alphas = [0.01, 0.1, 1, 10.0]
l1_ratios = [0.1, 0.5, 0.9]
C_values = [0.01, 0.1, 1, 10, 100]        # for SVR
kernel_values = ['linear', 'rbf', 'poly'] # for SVR
gamma_values = [0.01, 0.1, 1.0]           # for SVR
max_depth = [5, 10, 20]

# regr, regr_name, mse = qf.train_linear_regression(X_train, y_train, cv)
regr, regr_name, mse = qf.train_ridge_regression(X_train, y_train, alphas, cv)
# regr, regr_name, mse = qf.train_lasso_regression(X_train, y_train, alphas, cv)
# regr, regr_name, mse = qf.train_elastic_net(X_train, y_train, alphas, l1_ratios, cv)
# regr, regr_name, mse, best_params = train_svr(X_train, y_train, cv)
# regr, regr_name, mse = qf.train_random_forest(X_train, y_train, cv = cv)
# regr, regr_name, mse = train_gradient_boosting(X_train, y_train, cv=cv)


#%% Plot cross-validation results

# Calculate percentage of variation explained
cum_percent = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(f'Cumulative percentage variation explained by each successive PC in {regr_name}: {cum_percent}')

# Scale the PCA-transformed training and testing data
X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

# Fit the regression model on reduced data
regr.fit(X_train_pca_scaled[:, :best_n_components], y_train)
result = permutation_importance(regr, X_test_pca_scaled, y_test, n_repeats=10, random_state=42, n_jobs=2)

# Calculate RMSE
pred = regr.predict(X_test_pca_scaled)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f'Root Mean Squared Error (RMSE) for {regr_name}: {rmse}')

sorted_idx = result.importances_mean.argsort()

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': np.array(mags.columns)[sorted_idx],
                                    'Importance': result.importances_mean[sorted_idx]})
# Sort the feature importances table by importance in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(mags.columns)[sorted_idx])
ax.set_title("Permutation Importance (test set)")
ax.grid(True)
plt.show()


#%% Visualize the predicted vs. actual redshift
results_df = pd.DataFrame([])
results_df['z_spec'] = y_test
results_df[f'z_phot_{regr_name}'] = pred
results_df[f'delta_z_{regr_name}'] = results_df['z_spec'] - results_df[f'z_phot_{regr_name}']

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
title = f'Predicted redshift comparison\nfrom PCA and {regr_name}\nfor {datasetname}'
qf.plot_z(y_test, pred, datasetname, ax=ax[0], title=title)
qf.plot_delta_z_hist(results_df[f'delta_z_{regr_name}'], datasetname, ax=ax[1])

print('\n===== Results summary =====')
print(f'{regr_name}, {datasetname}, n = {y.shape[0]}')
print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')
print(f'Optimal number of components: {best_n_components}')
print(f"Mean Squared Error (MSE) from Cross-Validation ({regr_name}): {mse}")
print(f'Root Mean Squared Error (RMSE) for {regr_name}: {rmse}')
print(f'Cumulative percentage variation explained by each successive PC in {regr_name}: {cum_percent}')
print(qf.metrics_table(np.array(y_test), pred, regr_name))
# Display the sorted feature importances table
print("Feature Importances (sorted by importance in descending order):")
print(feature_importances)

metrics_df = qf.metrics_table(np.array(y_test), pred, regr_name)

#%% Apply neural network to the standardised, PCA'd datasets

if apply_neural_network:
    # Set X and y for the neural network
    X = X_train_pca_scaled  # Input features (standardised PCA-transformed training data)
    y = np.array(y_train)   # Target variable (redshift)

    # Model parameters
    model_params = {
        'X': X,
        'y': y,
        'n': X.shape[1],                   # Number of input features (number of PCA components)
        'num_layers': 3,                   # Number of hidden layers in your neural network
        'hyperparameters': [100, 'relu'],  # Number of neurons and activation function for hidden layers
        'loss_metric': 'mean_squared_error',  # Loss function for training
        'evaluation_metrics': ['mae'],        # Evaluation metrics during training
        'opt': 'Adam',                        # Optimizer
        'kf': cv                              # Cross-validation settings
    }

    # Build neural network model
    model = qf.build_nn_model(**model_params)

    # Make predictions on the test data
    y_pred = model.predict(X_test_pca_scaled)  # Use standardized PCA-transformed test data

    print(qf.metrics_table(y_test, y_pred.flatten(), 'Neural network'))
    nn_metrics_df = qf.metrics_table(y_test, y_pred.flatten(), 'Neural network')
    y_test = np.array(y_test)
    results_df['z_phot_NN'] = y_pred

    results_df['delta_z'] = results_df['z_spec'] - results_df['z_phot_NN']
#%% Visualize the predicted vs. spectroscopic redshift
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.tight_layout()
    title = f'Predicted redshift comparison from Neural Network\nfor {datasetname}'
    qf.plot_z(y_test, y_pred.flatten(), datasetname, ax=ax[0], title=title)
    qf.plot_delta_z_hist(results_df['delta_z'], datasetname, ax=ax[1])
