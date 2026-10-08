import pandas as pd


def filter_reviews_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Filter the reviews data by dropping NaN rows, and filtering out short reviews.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the reviews data.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing the reviews data.
        int: The number of reviews in the filtered DataFrame.
    """
    # Create a copy of the DataFrame first
    df = df.copy()

    # sort by review date, after that slice the first 4000 reviews
    df = df.sort_values(by=["date_reviewed"], ascending=False).iloc[:1000]

    # slice last 4000 reviews
    # df = df.iloc[-5500:]

    # drop nan review rows
    df = df.dropna(subset=["review_text"])

    # filter out short reviews
    # print(f"Before filtering: {len(df)} reviews")
    short_mask = df["review_text"].apply(lambda row: len(row) > 50)
    df = df.loc[short_mask]

    # print(f"After filtering: {len(df)} reviews")
    df = df.reset_index(drop=True)
    print(f"df head: {df.head()}")

    return df, len(df)


def prepare_reviews_data_for_llm(
    df_review_text_series: pd.Series, delimiter="##"
) -> str:
    """Prepare the reviews data for LLM.

    This function takes a pandas Series containing review text and prepares it for LLM.
    It joins the review texts using a delimiter and adds prefix and suffix delimiters to the final string.

    Args:
        df_review_text_series (pd.Series): A pandas Series containing review text.
        delimiter (str, optional): The delimiter used to join the review texts. Defaults to "##".

    Returns:
        str: The prepared reviews data for review analysis by an LLM.
    """

    # get array of reviews
    input_data_for_llm = df_review_text_series.values
    input_data_for_llm = (delimiter).join(input_data_for_llm)
    input_data_for_llm = f"{delimiter}{input_data_for_llm}{delimiter}"
    return input_data_for_llm
