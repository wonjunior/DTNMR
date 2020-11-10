import re

def process(data, ids, mapping, transform, policy):
    """
    process(data, ids, mapping transform, policy) --> (result, result_set)

    Description
    ---
    Processing pipeline to transform feature values presented by data, filter the invalid
    ones and keep track these in a dict. Summary of the operation:

        data ──── [transform] ──── [policy] ──── mapping
                        └──── result    └──── result_set

    Parameters
    ---
    data : list of string
        The input array, it represent the values of a single feature. Each element in the list
        represent the feature value as a string. Data points can have more than one feature
        value, in this case they are separated by a pipe sign.
    ids: list
        The array of ids, it allows to find the identifier of a data point given its index.
    mapping: dict {id: index}
        Somewhat the opposite of ids. Difference being that it is used to keep track of the
        data points that are discarded. These are tagged by setting their indices to -1.
    transform: function(string) -> any
        The transformation to apply to the input list.
    policy: function(any) -> bool
        The filter function to determine if a data point is invalidated or not. The transform
        function is applied *before* the policy is being called.

    Returns
    ---
        (result, result_set)

    result: list of a list of feature values
        The input list mapped with the transform function. It has the same size as the input
        list, even if some points have been invalidated by policy. This is done in order to
        keep consistency of the indices accross all feature lists.
    result_set: set of feature
        Will appear in this set only the feature values that have been validated by policy.
    """
    result, result_set  = [], set()
    for i, x in enumerate(data):
        x = list(map(transform, x.split('|')))
        result.insert(i, x)
        if mapping[ids[i]] != -1 and all(map(policy, x)):
            result_set.update(x)
        else:
            mapping[ids[i]] = -1
    return result, result_set

def no_special_character(x):
    """
    Returns false if the input string
        * contains Thai, Japanese, Chinese or Korean characters;
        * is no longer than 25 characters;
        * doesn't contain spaces.
    """
    return re.search('\s', x) and not(len(x) > 25
        or re.search("[\u4e00-\u9FFF]|[\u3040-\u30ff]|[\uac00-\ud7a3]|[\u0E00-\u0E7F]", x))
