from vmlkit import utility as utl
import category_encoders as ce
import pandas as pd


def one_hot_encode(df, verbose=0, drop_invariant=False,
                   return_df=True, handle_missing='value',
                   handle_unknown='value', use_cat_names=True,
                   return_encoder=False):
    """
    Onehot (or dummy) coding for categorical features,
    produces one feature per category, each binary.
    """
    cat_cols = utl.get_categorical_columns(df)

    enc = ce.OneHotEncoder(verbose=verbose,
                           cols=cat_cols,
                           drop_invariant=drop_invariant,
                           return_df=return_df,
                           handle_missing=handle_missing,
                           handle_unknown=handle_unknown,
                           use_cat_names=use_cat_names)

    encoded = enc.fit_transform(df[cat_cols])

    df = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)

    if return_encoder:
        return df, enc

    return df


def ordinal_encode(df, verbose=0, mapping=None, cols=None,
                   drop_invariant=False, return_df=True,
                   handle_unknown='value', handle_missing='value',
                   return_encoder=False):
    """
    Encodes categorical features as ordinal.
    This method is so-called "label encoding."
    """
    cat_cols = utl.get_categorical_columns(df)

    enc = ce.ordinal.OrdinalEncoder(verbose=verbose,
                                    mapping=None,
                                    cols=cat_cols,
                                    drop_invariant=drop_invariant,
                                    return_df=return_df,
                                    handle_unknown=handle_unknown,
                                    handle_missing=handle_missing)

    encoded = enc.fit_transform(df[cat_cols])

    df = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)

    if return_encoder:
        return df, enc

    return df


def target_encode(df, y, verbose=0, cols=None,
                  drop_invariant=False,
                  return_df=True,
                  handle_missing='value',
                  handle_unknown='value',
                  min_samples_leaf=1,
                  smoothing=1.0,
                  return_encoder=False):
    cat_cols = utl.get_categorical_columns(df)
    print('\nBefore encoding:\n', df[cat_cols])

    enc = ce.target_encoder.TargetEncoder(verbose=verbose,
                                          cols=cat_cols,
                                          drop_invariant=drop_invariant,
                                          return_df=return_df,
                                          handle_missing=handle_missing,
                                          handle_unknown=handle_unknown,
                                          min_samples_leaf=min_samples_leaf,
                                          smoothing=smoothing)

    df[cat_cols] = enc.fit_transform(df[cat_cols], y)
    print('\nAfter encoding:\n', df[cat_cols])

    if return_encoder:
        return df, enc

    return df


def frequency_encode(train, test, y):
    cat_cols = utl.get_categorical_columns(train)
    print('\nBefore encoding:\n', train[cat_cols])

    for c in cat_cols:
        freq = train[c].value_counts()

        train[c] = train[c].map(freq)
        test[c] = test[c].map(freq)
