import numpy as np
import unittest
from _stabletrees import Splitter
from _stabletrees import ProbabalisticSplitter


class TestMSE(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([0,1,4])
        return X,y
    
   
    
    def test_init(self):
        X,y = self.get_testset_one()
        n = len(y)
        
        splitter = Splitter(1,n,0,False)
        
        any_split,feature_index, impurity, score, split_value = splitter.find_best_split(X,y)

        print(n)
        
        self.assertTrue(feature_index==0)
        self.assertTrue(impurity== np.mean((y - y.mean())**2))
        self.assertTrue(score==1/6)


    def test_init_prob(self):
        X,y = self.get_testset_one()
        n = len(y)
       
        splitter = ProbabalisticSplitter(1,n,0,False,0)
        
        any_split, feature_index, impurity, score, split_value = splitter.find_best_split(X,y)
        
        self.assertTrue(feature_index==0)
        self.assertTrue(impurity== np.mean((y - y.mean())**2))
        self.assertTrue(score==1/6)


if __name__ == '__main__':
    unittest.main()
