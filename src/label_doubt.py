import numpy as np

def compute_threshold(y_true: np.array, preds: np.array) -> dict:
    """ Compute the average confidence of each class
    Params:
      y_true (np.array): Array of labels for predictions shape[num_example]
      preds (np.array): Model softmax predictions. shape[num_example, num_classes]
    Return:
      class_preds (dict): The keys are class numbers and 
                          contains [average confidence, indexes, predictions] of each class
    """
    class_preds = {}
    for i in range(preds.shape[1]):
      index = np.where(y_true == i)[0]
      threshold = np.mean(preds[index, i])
      class_preds[i] = {'threshold': threshold, 'indexes': index, 'preds': preds[index]}
    return class_preds

def find_doubt_ids(y_true: np.array, preds: np.array, margin: float=0.0) -> dict:
    """ Find the indexes of doubtable labels in the dataset
    Params:
      y_true (np.array): Array of labels for predictions shape[num_example]
      preds (np.array): Model softmax predictions. shape[num_example, num_classes]
      margin (float): Minimum distance from the average confidence of classes
    Return:
      doubts (dict): The keys are class numbers where values are 
                     [doubtable label ids, distance from average confidence]
    """
    classes = compute_threshold(y_true, preds)
    matrix = np.zeros((len(classes), len(classes)))
    doubts = {}
    for c1 in range(len(classes)):
        P = classes[c1]['preds']
        T = classes[c1]['threshold']
        ids = classes[c1]['indexes']
        for c2 in range(len(classes)):
          if c1 == c2: continue
          class_doubt_ids = np.where(P[:, c2] > (T + margin))[0]
          distances = P[class_doubt_ids, c2] - T
          if len(class_doubt_ids) > 0:
            doubts[c1] = {'ids': ids[class_doubt_ids].tolist(), 'distances': distances.tolist()}
            matrix[c1][c2] = len(class_doubt_ids)
    return doubts

  
if __name__ == '__main__':
    doubts = find_doubt_ids(y_test, preds)
    print(doubts)
