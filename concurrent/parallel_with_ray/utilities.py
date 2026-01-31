import ray
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import time
import matplotlib.pyplot as plt


class DataPreprocessor:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_full: pd.DataFrame = None

    def load_kaggle_dataset(self):
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        file_path = "retail_sales_dataset.csv"

        self.df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "terencekatua/retail-sales-dataset",
            file_path,
        )

        return self.df

    def create_full_date_index(self, check=True):
        import pandas as pd
        df1 = self.df.copy()
        df1["Date"] = pd.to_datetime(df1["Date"])

        # create full date index
        full_dates = pd.date_range(
            start=df1["Date"].min(),
            end=df1["Date"].max(),
            freq="D"
        )
        self.df_full = (
            df1.groupby(['Date', 'Product Category'], as_index=False).agg({'Quantity': 'sum'})
            .set_index("Date")
            .groupby("Product Category", group_keys=False)
            .apply(
                lambda x: (
                    x.reindex(full_dates)
                     .assign(**{"Product Category": x.name})
                ), include_groups=False
            )
            .reset_index()
            .rename(columns={"index": "Date"})
        )

        self.df_full["Quantity"] = self.df_full["Quantity"].fillna(0)
        if check:
            # sanity check: each category has full coverage
            assert self.df_full['Product Category'].value_counts().nunique() == 1
        return self.df_full

    def visualise_sales(self):
        import matplotlib.pyplot as plt
        import math

        if self.df_full is None:
            raise ValueError("Call create_full_date_index() first")

        categories = self.df_full["Product Category"].unique()
        n_cat = len(categories)
        n_cols = 3
        n_rows = math.ceil(n_cat / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 3 * n_rows),
            sharex=True
        )

        axes = axes.flatten()

        for ax, cat in zip(axes, categories):
            df_cat = self.df_full[self.df_full["Product Category"] == cat]
            ax.plot(df_cat["Date"], df_cat["Quantity"])
            ax.set_title(f"{cat} sales")
            ax.set_ylabel("Quantity")

        for ax in axes[n_cat:]:
            ax.axis("off")

        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.show()


class FeatureEngineer:
    REQUIRED_COLUMNS = {'Product Category', 'Date', 'Quantity'}

    def __init__(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.copy()
        self.keys = self.df['Product Category'].unique()

    def create_lag_features(self, lags):
        df = (
            self.df
            .sort_values(['Product Category', 'Date'])
            .reset_index(drop=True)
        )

        g = df.groupby('Product Category')['Quantity']

        for lag in lags:
            df[f'quantity_lag_{lag}'] = g.shift(lag)

            df[f'quantity_roll_mean_{lag}'] = (
                g.shift(1)
                .groupby(df['Product Category'])
                .rolling(lag)
                .mean()
                .reset_index(level=0, drop=True)
            )
            if lag >= 2:
                df[f'quantity_roll_std_{lag}'] = (
                    g.shift(1)
                    .groupby(df['Product Category'])
                    .rolling(lag)
                    .std()
                    .reset_index(level=0, drop=True)
                )
        self.df = df.dropna(how='any')

        return self.df

    def create_temporal_features(
            self,
            add_fourier=True,
            fourier_period=365,
            fourier_order=3):

        df = self.df.copy()

        # Basic calendar features
        df['dow'] = df['Date'].dt.weekday        # 0–6
        df['dom'] = df['Date'].dt.day            # 1–31
        df['doy'] = df['Date'].dt.dayofyear      # 1–365/366
        df['week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['month'] = df['Date'].dt.month

        # Fourier seasonality
        if add_fourier:
            t = df['doy']
            for k in range(1, fourier_order + 1):
                df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * t / fourier_period)
                df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * t / fourier_period)

        self.df = df
        return df

    def create_X_y(
            self,
            target_col='Quantity',
            dropna=False):
        """
        Returns:
            dict[product_category] = {
                'X': DataFrame,
                'y': Series
            }
        """
        results = {}

        for key in self.keys:
            df_cat = (
                self.df[self.df['Product Category'] == key]
                .sort_values('Date')
                .reset_index(drop=True)
            )

            if dropna:
                df_cat = df_cat.dropna()

            X = df_cat.drop(
                columns=['Product Category', 'Date', target_col],
                errors='ignore'
            )

            # replace NaN with mode (column-wise)
            modes = X.mode(dropna=True)
            if not modes.empty:
                X = X.fillna(modes.iloc[0])

            y = df_cat[target_col]

            results[key] = {
                'X': X,
                'y': y
            }

        return results


def forecast_one_item(category, data_dict, plot=True):
    X, y = data_dict[category]['X'], data_dict[category]['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train, y_train)

    train_preds = lr.predict(X_train)
    test_preds = lr.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)

    result = pd.DataFrame(data=[test_preds, y_test], index=['test_preds', 'y_test'])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Full history with train/test split
        ax1 = axes[0]
        train_indices = y_train.index
        test_indices = y_test.index

        ax1.plot(
            train_indices,
            y_train,
            label='Train (Actual)',
            color='blue',
            alpha=0.7)
        ax1.plot(test_indices, y_test, label='Test (Actual)', color='green', alpha=0.7)
        ax1.plot(test_indices, test_preds, label='Test (Forecast)',
                 color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=train_indices[-1], color='black', linestyle=':',
                    label='Train/Test Split', linewidth=1.5)
        ax1.set_title(f'{category} - Historical & Forecast')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Quantity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Test period only (zoomed in)
        ax2 = axes[1]
        ax2.plot(test_indices, y_test, label='Actual',
                 marker='o', color='green', linewidth=2)
        ax2.plot(test_indices, test_preds, label='Forecast',
                 marker='s', color='red', linestyle='--', linewidth=2)
        ax2.set_title(f'{category} - Test Period Detail (MAE: {test_mae:.2f})')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Quantity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    time.sleep(10)
    print(f'{category} mae is {test_mae:.2f}')
    return category, result, test_mae


@ray.remote
def forecast_one_item_ray(category, data_dict, plot=False):
    X, y = data_dict[category]['X'], data_dict[category]['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train, y_train)

    train_preds = lr.predict(X_train)
    test_preds = lr.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)

    result = pd.DataFrame(data=[test_preds, y_test], index=['test_preds', 'y_test'])

    # Note: Plotting from Ray workers can be tricky, so disabled by default
    # If you want plots, consider plotting after collecting results
    if plot:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for Ray workers
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Full history with train/test split
        ax1 = axes[0]
        train_indices = y_train.index
        test_indices = y_test.index

        ax1.plot(
            train_indices,
            y_train,
            label='Train (Actual)',
            color='blue',
            alpha=0.7)
        ax1.plot(test_indices, y_test, label='Test (Actual)', color='green', alpha=0.7)
        ax1.plot(test_indices, test_preds, label='Test (Forecast)',
                 color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=train_indices[-1], color='black', linestyle=':',
                    label='Train/Test Split', linewidth=1.5)
        ax1.set_title(f'{category} - Historical & Forecast')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Quantity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Test period only
        ax2 = axes[1]
        ax2.plot(test_indices, y_test, label='Actual',
                 marker='o', color='green', linewidth=2)
        ax2.plot(test_indices, test_preds, label='Forecast',
                 marker='s', color='red', linestyle='--', linewidth=2)
        ax2.set_title(f'{category} - Test Period Detail (MAE: {test_mae:.2f})')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Quantity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'/tmp/{category}_forecast.png', dpi=100, bbox_inches='tight')
        plt.close()

    time.sleep(10)
    print(f'{category} mae is {test_mae:.2f}')
    return category, result, test_mae


def print_time(label: str, seconds: float, colour: str = "green"):
    colours = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    c = colours.get(colour, "")
    r = colours["reset"]

    print(f"\n{c}{'=' * 10} {label}: {seconds:.2f} seconds {'=' * 10}{r}\n")
