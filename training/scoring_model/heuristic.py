class HeuristicScoringModel:
    def predict(self, distance_km: float, remaining_capacity_ton: float, cargo_weight_ton: float) -> tuple[float, str]:
        # Cek jika kargo lebih berat dari sisa kapasitas kapal
        if cargo_weight_ton > remaining_capacity_ton:
            return 0.0, "Kapasitas kargo melebihi sisa kapasitas kapal."
        
        # Hitung skor utilitas
        utilization_score = cargo_weight_ton / remaining_capacity_ton
        
        # Hitung skor jarak (penalti maksimal di 1000 km)
        distance_score = max(0.0, 1.0 - (distance_km / 1000.0))

        # Bobot tertimbang
        final_score = (utilization_score * 0.6) + (distance_score * 0.4)
        explanation = f"Heuristic Mode -> Utilitas: {utilization_score:.2f}, Jarak: {distance_score:.2f}"
        
        return min(round(final_score, 2), 1.0), explanation