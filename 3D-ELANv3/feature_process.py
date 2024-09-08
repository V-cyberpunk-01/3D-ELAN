import pandas as pd
from xenonpy.descriptor import Compositions
from xenonpy.datatools import preset
from sklearn.decomposition import PCA
# preset.sync('elements')
# preset.sync('elements_completed')

pca = PCA(n_components=64)

def compFeatureHelper():
    raw = pd.read_csv('0916/composition/b.csv').astype('float64')
    eleName = raw.columns
    comp = pd.DataFrame(raw, columns=eleName).values
    compObj = []
    for i in range(len(raw)):
        tmp = {}
        for j in range(len(eleName)):
            if comp[i][j] != 0:
                tmp[eleName[j]] = comp[i][j]
        compObj.append(tmp)

    cal = Compositions()
    compRes = cal.transform(compObj)

    print(compRes['sum:atomic_number'])
    # 290 feature
    print(len(compRes.columns))
    features_list = compRes.columns.tolist()[94:]
    print(features_list)
    compFeatures = pd.DataFrame(compRes, columns=features_list).dropna(0)
    newFeatures = pd.DataFrame(pca.fit_transform(compFeatures))

    newFeatures.to_csv('0916/composition/composition_feature_b_pca.csv')
    return newFeatures

result = compFeatureHelper()
print(result)