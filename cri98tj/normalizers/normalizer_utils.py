"""
ritorna un dataset nella forma class, columns_n... con tid come index
"""
def dataframe_pivot(df, maxLen, verbose, fillna_value, columns):
    df["pos"] = df.groupby(['tid', 'partId']).cumcount()

    if maxLen is not None:
        if maxLen >= 1:
            if verbose: print(F"Cutting sub-trajectories length at {maxLen} over {df.max().pos}", flush=True)
            df = df[df.pos < maxLen]
        else:
            if verbose: print(F"Cutting sub-trajectories length at {df.quantile(.95).pos} over {df.max().pos}", flush=True)
            df = df[df.pos < df.quantile(.95).pos]

    if verbose: print("Pivoting tables", flush=True)
    df_pivot = df.groupby(['partId', 'pos'])[columns].max().unstack().reset_index()
    df_pivot = df_pivot.merge(df.groupby(['partId'])['class'].max().reset_index(), on=["partId"])
    #df_pivot["size"] = df.groupby(['tid']).size()
    df_pivot = df_pivot.drop(columns=[("partId", "")]).set_index("partId")

    if fillna_value is not None:
        df_pivot.fillna(fillna_value, inplace=True)

    return df_pivot[["class"]+[x for x in df_pivot.columns if x != "class"]]