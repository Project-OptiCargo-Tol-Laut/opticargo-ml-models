import statistics

class HeuristicAnomalyModel:
    def predict(self, unit_price: float, historical_prices: list[float]) -> tuple[bool, str]:
        if not historical_prices:
            return False, "Data histori harga kosong."
        
        median_price = statistics.median(historical_prices)
        
        # Sesuai PRD: Aturan threshold sederhana (misal > 3x median dianggap anomali)
        is_anomaly = unit_price > (3 * median_price) or unit_price < (0.2 * median_price)
        
        explanation = f"Median harga: {median_price:.2f}. Harga diuji: {unit_price:.2f}."
        if is_anomaly:
            explanation += " (Terdeteksi Anomali: Harga berada di luar batas wajar)."
        else:
            explanation += " (Harga Normal)."
            
        return is_anomaly, explanation