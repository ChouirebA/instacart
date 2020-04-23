from Read_file import *

def final_table():
    nb_users = 10000
    df = user_table(nb_users)
    maxlen = 12
    T = 5
    panier = 21

    X = np.array(df)
    X = np.zeros((nb_users, T, maxlen))
    for i in range(len(df)):
        xi = pad_sequences(sequences=df[i], maxlen=maxlen, dtype='int32', padding='pre', truncating='pre', value=0)
        t = min(T, xi.shape[0])
        xi = xi[:t]
        a = T - t
        X[i][a:] = xi

    data = []
    label = []

    for user in X:
        #         print("USER:", user)
        #         print("\n")
        liste = []
        for order in user:
            # print("ici c order", order)
            p = np.zeros(panier + 1)
            for department in order:
                # print("ici c departement", department)
                # print("\n")
                x = int(department)
                if x == 0:
                    p[x] = 0
                else:
                    p[x] = 1

            liste.append(list(p))
        #             print("ici on fait la liste", liste)
        #             print("\n")

        X_encoded = liste[:][:-1][:]
        #         print("ici c data", X_encoded)
        #         print('\n')
        data.append(X_encoded)
        #         print("ici c data", data)
        #         print('\n')

        Y_encoded = liste[:][-1][:]
        #         print("ici c Y_encoded", Y_encoded)
        #         print('\n')
        label.append(Y_encoded)

    data = np.array(data)
    label = np.array(label)

    return data, label
