class HeuristicForecastingModel:
    def predict(self, historical_volumes_ton: list[float]) -> tuple[float, str]:
        if not historical_volumes_ton:
            return 0.0, "Data histori kosong."
        
        # Sesuai PRD: Menghitung rata-rata bergerak sederhana (Simple Moving Average)
        forecast_val = sum(historical_volumes_ton) / len(historical_volumes_ton)
        return round(forecast_val, 2), f"Moving average dari {len(historical_volumes_ton)} titik data histori."