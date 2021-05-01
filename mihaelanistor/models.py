
# todo add source
def NearestNeighbor(distmat, labels):
    test_labels = []
    predicted_labels = []
    counter = 0
    for i in range(len(labels)):
        d_temp = list(distmat[i,:])
        l_temp = labels.copy()
        d_temp.pop(i)
        l_temp.pop(i)

        test_labels.append(int(labels[i]))

        mn,idx = min( (d_temp[i],i) for i in range(len(d_temp)) )
        predicted_labels.append(int(l_temp[idx]))
        if int(labels[i]==l_temp[idx]):
            counter+=1

    return (counter*100/len(labels))