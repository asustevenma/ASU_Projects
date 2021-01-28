{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Import classifiers thay you want to use\n",
    "from sklearn import tree\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Import model tuning   methods\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n",
    "# Import evaluate methods that you want to use\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import roc_curve, auc , roc_auc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Read Data\n",
    "train_df = pd.read_csv(\"Desktop/Launch_Angle_Training.csv\")\n",
    "#test_df = pd.read_csv(\"Desktop/Test_data.csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(69914, 29)\n"
     ]
    }
   ],
   "source": [
    "# Drop every row with NAN\n",
    "train_clean = train_df.dropna()\n",
    "print(train_clean.shape)\n",
    "# Get statistics about the data(Optional)\n",
    "#print(train_clean.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Prepare trainning dataset and target dataset\n",
    "train_X = train_clean.drop('Outcome', axis = 1)\n",
    "train_Y = train_clean['Outcome']\n",
    "target = pd.factorize(train_clean['Outcome'])[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Convert text data(object type) into dummy variables\n",
    "train_convert = pd.get_dummies(data = train_X, columns=['pitch_type', 'stand', 'p_throws', 'bb_type'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(55931, 46) (13983, 46) (55931,) (13983,)\n"
     ]
    }
   ],
   "source": [
    "# Split dataset to train the model\n",
    "X_train, X_test, y_train, y_test = train_test_split(train_convert, target, test_size = 0.2, random_state=42, stratify = train_Y)\n",
    "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create Random Forest object and Initialize Random Forest Classsifer\n",
    "randomforest_model = RandomForestClassifier(n_estimators=1000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Steven Ma\\Anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_split.py:605: Warning: The least populated class in y has only 3 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.\n",
      "  % (min_groups, self.n_splits)), Warning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.95209495  0.95258867  0.95301437  0.95300765  0.95293276]\n"
     ]
    }
   ],
   "source": [
    "# Compute 5-fold cross-validation to examine the model\n",
    "cv_scores = cross_val_score(randomforest_model, train_convert, target, cv=5)\n",
    "print(cv_scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup the hyperparameter grid\n",
    "#param = dict(epochs=[10,20,30])\n",
    "\n",
    "# Hyperparameter tuning - instantiate the RandomizedSearchCV object: randomforest_cv\n",
    "#randomforest_cv = GridSearchCV(randomforest_model, param_grid = param, cv = 5, n_jobs=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
       "            max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit Training Data\n",
    "randomforest_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 0 ..., 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "# predict Output\n",
    "y_predict = randomforest_model.predict(X_test)\n",
    "print(y_predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score: 0.952799828363\n",
      "F1 Score: 0.931226546314\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Steven Ma\\Anaconda3\\lib\\site-packages\\sklearn\\metrics\\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    }
   ],
   "source": [
    "# Get accuracy score\n",
    "acc_score = accuracy_score(y_test, y_predict)\n",
    "print(\"Accuracy Score:\", acc_score)\n",
    "# Get F1 score\n",
    "f1 = f1_score(y_test, y_predict, average = 'weighted')\n",
    "print(\"F1 Score:\", f1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Predicted Outcomes</th>\n",
       "      <th>0</th>\n",
       "      <th>2</th>\n",
       "      <th>4</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Actual Outcomes</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13300</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>214</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>156</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>247</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>32</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Predicted Outcomes      0   2  4\n",
       "Actual Outcomes                 \n",
       "0                   13300   1  1\n",
       "1                     214   0  0\n",
       "2                     156  23  0\n",
       "3                     247   1  0\n",
       "4                      32   2  0\n",
       "5                       5   0  0\n",
       "6                       1   0  0"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create confusion matrix\n",
    "pd.crosstab(y_test, y_predict, rownames=['Actual Outcomes'], colnames=['Predicted Outcomes'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# predict probabilities(for ROC Curve)\n",
    "prob = randomforest_model.predict_proba(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.90486484945526424"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Calculate ROC and AUC\n",
    "fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1] , pos_label = 1)\n",
    "aucc = auc(fpr, tpr)\n",
    "aucc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nFO4iIlno/wMCzCz8dghKIgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1f0c48695c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot ROC\n",
    "plt.plot(fpr,tpr)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
