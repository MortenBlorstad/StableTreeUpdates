import numpy as np
import unittest
from _stabletrees import MSE, Poisson, MSEReg, PoissonReg, MSEABU, PoissonABU


class TestMSE(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([0,1,4])
        return X,y



    def test_init(self):
        X,y = self.get_testset_one()
        n = len(y)
        mse = MSE()
        mse.init(n,y)
        self.assertTrue(mse.get_score()==0)
        self.assertTrue(mse.node_impurity(y)== np.mean((y - y.mean())**2))

    def test_update(self):
        X,y = self.get_testset_one()
        mse = MSE()
        n = len(y)
        mse.init(n,y)
        correct_scores = [1.5,0.1667]
        for i, y_i in enumerate(y):
            mse.update(y_i)
            score = np.round(mse.get_score(),4)
            if i <len(correct_scores):
                self.assertTrue(score ==correct_scores[i])


    # def test_init_with_reg(self):
    #     X,y = self.get_testset_one()
    #     yprev = np.array([0.25,0.25,4])
    #     n = len(y)
    #     mse = MSEReg()

    #     mse.init(n,y,yprev)
    #     self.assertTrue(mse.get_score()==0)
    #     self.assertTrue(mse.node_impurity(y)== np.mean((y - y.mean())**2))

    # def test_update(self):
    #     X,y = self.get_testset_one()
    #     yprev = np.array([0.25,0.25,4])
    #     mse = MSEReg()
    #     n = len(y)
    #     mse.init(n,y,yprev)
    #     correct_scores = [round(1.5+2.4583,4), round(0.16667+0.0416667,4)]
    #     for i, (y_i,yp_i) in enumerate(zip(y,yprev)):
    #         mse.update(y_i,yp_i)
    #         score = np.round(mse.get_score(),4)
    #         if i <len(correct_scores):
    #             self.assertTrue(score == correct_scores[i])


class TestMSEABU(unittest.TestCase):
    def get_testset_one(self):
        np.random.seed(0)
        X =np.random.uniform(size=(10,1), low = 0,high = 1)
        y = np.random.normal(X.ravel())
        return X,y



    def test_init(self):
        X,y = self.get_testset_one()
        yb = np.random.choice(y, size = 10, replace=True)
        w = np.ones(20)
        w[10:] = 0.5
        y_ = np.concatenate((y,yb))
        n = len(y_)
        mse = MSEABU(1)
        mse.init(n,y_,w)
        
        self.assertTrue(mse.node_impurity(y_) -  np.mean((y_ - y_.mean())**2)<0.00000001)

    def test_update(self):
        X,y = self.get_testset_one()
        yb = np.random.choice(y, size = 10, replace=True)
        w = np.ones(20)
        w[10:] = 0.5
        y_ = np.concatenate((y,yb))
        n = len(y_)
        mse = MSEABU(1)
        mse.init(n,y_,w)
        y_sum_l = 0
        sum_w_l = 0
        sum_wxy_l =0
        y_sum_r = sum(y_)
        sum_w_r = sum(w)
        sum_wxy_r =sum(y_*w)
        nl = 0
        nr = n
        for i, (y_i,w_i) in enumerate(zip(y_[:-3],w[:-3])):
            mse.update(y_i,w_i)
            score = mse.get_score()
            nl+=1
            nr-=1
            y_sum_l+=y_i
            sum_w_l+=w_i
            sum_wxy_l+= w_i*y_i
            y_sum_r-=y_i
            sum_w_r-=w_i
            sum_wxy_r-= w_i*y_i
            # print(y_sum_l,sum_w_l,sum_wxy_l,nl, 2*sum_wxy_l*(y_sum_l/nl) )
            # print(y_sum_r,sum_w_r,sum_wxy_r,nr, 2*sum_wxy_r*(y_sum_r/nr)  )
            red = (np.sum(w[:(i+1)]*(y_[:(i+1)] - y_[:(i+1)].mean())**2 ) + np.sum(w[(i+1):]*(y_[(i+1):] - y_[(i+1):].mean())**2 ))/20
            # print(score, red,   2*sum(w[:(i+1)]*y_[:(i+1)])*np.mean(y_[:i]), 2*sum(w[(i+1):]*y_[(i+1):])*np.mean(y_[(i+1):]))
            # print(np.mean(y_[:(i+1)]), np.mean(y_[(i+1):]))
            # print(i)
            # print( (sum(w*(y_**2)) - 2*sum(w[:(i+1)]*y_[:(i+1)])*np.mean(y_[:(i+1)]) + sum(w[:(i+1)])*np.mean(y_[:(i+1)])**2  - 2*sum(w[(i+1):]*y_[(i+1):])*np.mean(y_[(i+1):]) + sum(w[(i+1):])*np.mean(y_[(i+1):])**2  )/20 )
            self.assertTrue(abs(score -  red)   <0.00000001)


class TestPoissinABU(unittest.TestCase):
    def get_testset_one(self):
        np.random.seed(0)
        X =np.random.uniform(size=(10,1), low = 0,high = 1)
        y = np.exp(np.random.normal(X.ravel()))
        return X,y



    def test_init(self):
        X,y = self.get_testset_one()
        yb = np.random.choice(y, size = 10, replace=True)
        w = np.ones(20)
        w[10:] = 0.5
        y_ = np.concatenate((y,yb))
        n = len(y_)
        mse = PoissonABU()
        mse.init(n,y_,w)
        self.assertTrue(mse.node_impurity(y_) -  2*np.mean(y_*np.log(y_/np.mean(y_))- (y_-np.mean(y_)))<0.00000001)

    def test_update(self):
        X,y = self.get_testset_one()
        index = np.random.choice(np.arange(0,10), size = 10, replace=True)
        Xb = X[index,:]
        yb = y[index]
        
        w = np.zeros(20)
        w[10:] = 0.5
        X_ = np.vstack((X,Xb))
        index = np.argsort(X_[:,0])
        y_ = np.concatenate((y,yb))
        n = len(y_)
        y_ = y_[index]
        w = w[index]
        mse = PoissonABU()
        mse.init(n,y_,w)
        y2_sum_l = 0
        y_sum_l = 0
        sum_w_l = 0
        sum_wxy_l =0
        y_sum_r = sum(y_)
        y2_sum_r = sum(y_[w==0])
        sum_w = sum(w)
        sum_w_r = sum_w
        sum_wxy_r =sum(np.log(y_)*w)
        nl = 0
        nr = n
        n2_l =0
        n2_r =10
        eps = 0.0000001
        y_sum_squared = np.sum((np.log(y_+eps)**2)*w)
        
        print(w)
        print( np.sum(y.mean()-np.log(y.mean())*y)  +  np.sum((np.zeros(10)+0.5)*(yb  - y.mean())**2 ))
        for i, (y_i,w_i) in enumerate(zip(y_[:-1],w[:-1])):
            mse.update(y_i,w_i)
            score = mse.get_score()
            if w_i ==0:
                n2_l +=1
                n2_r -=1
                y2_sum_l+=y_i
                y2_sum_r -= y_i
            nl+=1
            nr-=1
            y_sum_l+=y_i
            y_sum_r-=y_i
           
            sum_wxy_l+= w_i*np.log(y_i+eps)
            sum_wxy_r-= w_i*np.log(y_i+eps)
            sum_w_l+=w_i
            sum_w_r-=w_i
            
            ind = i+1
            mask1 = w[:ind] ==0
            mask2 = w[ind:] ==0

            print(y_sum_squared,y2_sum_l,y2_sum_r,ind)     
            print(sum_w, sum_wxy_l,sum_wxy_r)  
            print("left: ", np.sum(y_[:ind][mask1].mean()-np.log(y_[:ind][mask1].mean())*y_[:ind][mask1]), np.sum(0.5*(y_[:ind][mask1]  - y_[:ind][mask1].mean())**2 ))  
            pois = np.sum(np.mean(y_[:ind][mask1])-y_[:ind][mask1]*np.log(np.mean(y_[:ind][mask1]))) + np.sum(np.mean(y_[ind:])  -y_[ind:][mask2]*np.log(np.mean(y_[ind:]) ))
            reg = np.sum(w[:ind][~mask1]*(y_[:ind][~mask1] - y_[:ind][mask1].mean())**2 ) + np.sum(w[ind:][~mask2]*(y_[ind:][~mask2] - y_[ind:][mask2].mean())**2 )
            red = (pois+reg)/20
            print("pois: ",pois, ", reg: ",reg)
            print(score,red)
            print((y_[:ind][mask1]+eps).mean(), np.sum(np.log(y_[:ind][mask1].mean())*y_[:ind][mask1]))
            print((y_[ind:][mask2]+eps).mean(), np.sum(np.log(y_[ind:][mask2].mean())*y_[ind:][mask2]))
            print(y2_sum_l - np.log(y2_sum_l/n2_l)*y2_sum_l +y2_sum_r - np.log(y2_sum_r/n2_r)*y2_sum_r )
            SSL = sum_w_l*(np.log(y2_sum_l/n2_l)**2)
            SSR = sum_w_r*(np.log(y2_sum_r/n2_r)**2)
            print("SS ",SSL, SSR, np.sum(w[:ind][~mask1]*(np.log(y_[:ind][mask1].mean())**2) ) )
            print(2*sum_wxy_l*np.log(y2_sum_l/n2_l), 2*sum_wxy_r*np.log(y2_sum_r/n2_r))
            print("reg ",y_sum_squared - 2*sum_wxy_l*np.log((y2_sum_l/n2_l) + eps)- 2*sum_wxy_r*np.log((y2_sum_r/n2_r)+eps) + SSL+SSR)
            print("SS left and right: ", len(y_[:ind][mask1]),y_sum_squared, np.sum(w[:ind][~mask1]*(np.log(y_[:ind][~mask1])**2) ) , np.sum(w[ind:][~mask2]*(np.log(y_[ind:][~mask2])**2) ) )
            print("SSL and SSR: ",len(y_[:ind][mask1]), np.sum(w[:ind][~mask1]*(np.log(y_[:ind][mask1].mean())**2) ) ,np.sum(w[ind:][~mask2]*np.log(y_[ind:][mask2].mean())**2) ) 
            print("yxpred: ", -2*np.sum(w[:ind][~mask1]*np.log(y_[:ind][~mask1])*np.log(y_[:ind][mask1].mean())) )
            print("yxpred: ", -2*np.sum(w[ind:][~mask2]*np.log(y_[ind:][~mask2])*np.log(y_[ind:][mask2].mean())) )
            print("yxpred: ", np.sum(w[:ind][~mask1]*(np.log(y_[:ind][~mask1].mean()) -  np.log(y_[:ind][mask1].mean()))**2 ) -  y_sum_squared - np.sum(w[:ind][~mask1]*(np.log(y_[:ind][mask1].mean())**2) ))
            print((np.sum(0.5*(np.log(y_[:ind][~mask1].mean()) -  np.log(y_[:ind][mask1].mean().mean()))**2 ) ) - np.sum((np.log(y_[:ind][mask1].mean())**2) ) - np.sum(w[:ind][~mask1]*(np.log(y_[:ind][mask1].mean())**2) )) 
            print(len(y_[:ind][mask1]), len(y_[ind:][mask2]), (np.sum(y_[:ind][mask1].mean()-np.log(y_[:ind][mask1].mean())*y_[:ind][mask1])  +  np.sum(y_[ind:][mask2].mean()-np.log(y_[ind:][mask2].mean())*y_[ind:][mask2])) )
            print(len(y_[:ind][mask1]), len(y_[ind:][mask2]), np.sum(w[:ind][~mask1]*(np.log(y_[:ind][~mask1]) -  np.log(y_[:ind][mask1].mean()))**2 )  ,  np.sum(w[ind:][~mask2]*(np.log(y_[ind:][~mask2]) -  np.log(y_[ind:][mask2].mean()))**2 ))
            print(np.sum(w[:ind][~mask1]*(np.log(y_[:ind][~mask1]) - np.log(y_[:ind][mask1].mean()))**2 ) + np.sum(w[ind:][~mask2]*(np.log(y_[ind:][~mask2]) - np.log(y_[ind:][mask2].mean()))**2 ))
            print()
            #self.assertTrue( abs(score -  red)   <0.00000001)
    

if __name__ == '__main__':
    unittest.main()