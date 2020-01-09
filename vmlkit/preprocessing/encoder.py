from vmlkit import utility as utl
import category_encoders as ce
import pandas as pd


def one_hot_encode(df, verbose=0, drop_invariant=False,
                   return_df=True, handle_missing='value',
                   handle_unknown='value', use_cat_names=True,
                   need_encoder=False):
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

    df_ = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)

    if need_encoder:
        return df_, enc

    return df_


def ordinal_encode(df, verbose=0, mapping=None, cols=None,
                   drop_invariant=False, return_df=True,
                   handle_unknown='value', handle_missing='value',
                   need_encoder=False):
    """
    Encodes categorical features as ordinal, in one ordered feature.
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

    df_ = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)

    if need_encoder:
        return df_, enc

    return df_


def target_encode(df, verbose=0, cols=None,
                  drop_invariant=False,
                  return_df=True,
                  handle_missing='value',
                  handle_unknown='value',
                  min_samples_leaf=1,
                  smoothing=1.0,
                  need_encoder=False):
    """
    Encodes categorical features as ordinal, in one ordered feature.
    This method is so-called "label encoding."
    """
    cat_cols = utl.get_categorical_columns(df)

    enc = ce.ordinal.OrdinalEncoder(verbose=verbose, cols=cols,
                                    drop_invariant=drop_invariant,
                                    return_df=return_df,
                                    handle_missing=handle_missing,
                                    handle_unknown=handle_unknown,
                                    min_samples_leaf=min_samples_leaf,
                                    smoothing=smoothing)

    encoded = enc.fit_transform(df[cat_cols])

    df_ = pd.concat([df.drop(cat_cols, axis=1), encoded], axis=1)

    if need_encoder:
        return df_, enc

    return df_
