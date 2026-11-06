def p50_ranking(per_od, labels=None):
    if per_od.duplicated(["algorithm", "od_id"]).any():
        raise ValueError("Повторяющиеся маршрут и пара")
    values = per_od.pivot(index="od_id", columns="algorithm", values="counted").fillna(0).mean()
    result = values.rename("P50").to_frame().sort_values("P50", ascending=False, kind="stable")
    result.insert(0, "место", result.P50.rank(ascending=False, method="min").astype("Int64"))
    return result.rename(index=labels).rename_axis("команда") if labels else result
