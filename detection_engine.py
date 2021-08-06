# Basic libraries
import collections
import numpy as np
from sklearn.ensemble import IsolationForest
from surpriver.data_loader import DataEngine
import warnings

warnings.filterwarnings("ignore")

class Surpriver:
    def __init__(self, top_n, history_to_use, min_volume, data_dictionary_path, data_granularity_minutes,
                 output_format, volatility_filter, stock_list, data_source, logger_queue):
        self.TOP_PREDICTIONS_TO_PRINT = top_n
        self.HISTORY_TO_USE = history_to_use
        self.MINIMUM_VOLUME = min_volume
        self.IS_LOAD_FROM_DICTIONARY = 1
        self.DATA_DICTIONARY_PATH = data_dictionary_path
        self.IS_SAVE_DICTIONARY = 0
        self.DATA_GRANULARITY_MINUTES = data_granularity_minutes
        self.IS_TEST = 0
        self.FUTURE_BARS_FOR_TESTING = 0
        self.VOLATILITY_FILTER = volatility_filter
        self.OUTPUT_FORMAT = output_format
        self.STOCK_LIST = stock_list
        self.DATA_SOURCE = data_source
        self._logger_queue = logger_queue

        self._logger_queue.put(["DEBUG", f" Surpriver: Surpriver has been initialized"])

        # Create data engine
        self.dataEngine = DataEngine(self.HISTORY_TO_USE, self.DATA_GRANULARITY_MINUTES,
                                     self.IS_SAVE_DICTIONARY, self.IS_LOAD_FROM_DICTIONARY, self.DATA_DICTIONARY_PATH,
                                     self.MINIMUM_VOLUME,
                                     self.IS_TEST, self.FUTURE_BARS_FOR_TESTING,
                                     self.VOLATILITY_FILTER,
                                     self.STOCK_LIST,
                                     self.DATA_SOURCE, self._logger_queue)

    def parse_large_values(self, value):
        if value < 1000:
            value = str(value)
        elif value >= 1000 and value < 1000000:
            value = round(value / 1000, 2)
            value = str(value) + "K"
        else:
            value = round(value / 1000000, 1)
            value = str(value) + "M"

        return value

    def calculate_volume_changes(self, historical_price):
        volume = list(historical_price["Volume"])
        dates = list(historical_price["Datetime"])
        dates = [str(date) for date in dates]

        # Get volume by date
        volume_by_date_dictionary = collections.defaultdict(list)
        for j in range(0, len(volume)):
            date = dates[j].split(" ")[0]
            volume_by_date_dictionary[date].append(volume[j])

        for key in volume_by_date_dictionary:
            volume_by_date_dictionary[key] = np.sum(
                volume_by_date_dictionary[key])  # taking average as we have multiple bars per day.

        # Get all dates
        all_dates = list(reversed(sorted(volume_by_date_dictionary.keys())))
        latest_date = all_dates[0]
        latest_data_point = list(reversed(sorted(dates)))[0]

        # Get volume information
        today_volume = volume_by_date_dictionary[latest_date]
        average_vol_last_five_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:6]])
        average_vol_last_twenty_days = np.mean([volume_by_date_dictionary[date] for date in all_dates[1:20]])

        return latest_data_point, self.parse_large_values(today_volume), self.parse_large_values(
            average_vol_last_five_days), self.parse_large_values(average_vol_last_twenty_days)

    def calculate_recent_volatility(self, historical_price):
        close_price = list(historical_price["Close"])
        volatility_five_bars = np.std(close_price[-5:])
        volatility_twenty_bars = np.std(close_price[-20:])
        volatility_all = np.std(close_price)
        return volatility_five_bars, volatility_twenty_bars, volatility_all


    def find_anomalies(self):
        """
        Main function that does everything
        """

        # Gather data for all stocks
        if self.IS_LOAD_FROM_DICTIONARY == 0:
            features, historical_price_info, future_prices, symbol_names = self.dataEngine.collect_data_for_all_tickers()
        else:
            # Load data from dictionary
            features, historical_price_info, future_prices, symbol_names = self.dataEngine.load_data_from_dictionary()

        # Find anomalous stocks using the Isolation Forest model. Read more about the model at -> https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        detector = IsolationForest(n_estimators=100, random_state=0)
        detector.fit(features)
        predictions = detector.decision_function(features)

        # Print top predictions with some statistics
        predictions_with_output_data = [[predictions[i], symbol_names[i], historical_price_info[i], future_prices[i]]
                                        for i in range(0, len(predictions))]
        predictions_with_output_data = list(sorted(predictions_with_output_data))

        # Results object for storing results in JSON format
        results = []

        for item in predictions_with_output_data[:self.TOP_PREDICTIONS_TO_PRINT]:
            # Get some stats to print
            prediction, symbol, historical_price, future_price = item

            latest_date, today_volume, average_vol_last_five_days, average_vol_last_twenty_days = self.calculate_volume_changes(
                historical_price)
            volatility_vol_last_five_days, volatility_vol_last_twenty_days, _ = self.calculate_recent_volatility(
                historical_price)
            if average_vol_last_five_days == None or volatility_vol_last_five_days == None:
                continue

            if self.OUTPUT_FORMAT == "CLI":
                print(
                    "Last Bar Time: %s\nSymbol: %s\nAnomaly Score: %.3f\nToday Volume: %s\nAverage Volume 5d: %s\nAverage Volume 20d: %s\nVolatility 5bars: %.3f\nVolatility 20bars: %.3f\n----------------------" %
                    (latest_date, symbol, prediction,
                     today_volume, average_vol_last_five_days, average_vol_last_twenty_days,
                     volatility_vol_last_five_days, volatility_vol_last_twenty_days))

            results.append(symbol)

        return results
