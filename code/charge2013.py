# -*- coding: utf-8 -*-

import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp

Protein_Atom = ['C','N','O','S','H']
Ligand_Atom = ['C','N','O','S','H','P','F','Cl','Br','I']
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']


Year = '2013'
pre = '../data/' + Year + '/'

f1 = open(pre + 'name/train_data_' + Year + '.txt')
pre_train_data = f1.readlines()
train_data = eval(pre_train_data[0])
f1.close()

f1 = open(pre + 'name/test_data_' + Year + '.txt')
pre_test_data = f1.readlines()
test_data = eval(pre_test_data[0])
f1.close()

f1 = open(pre + 'name/all_data_' + Year + '.txt')
pre_all_data = f1.readlines()
all_data = eval(pre_all_data[0])
f1.close()




########################################################################################
# extract coordinate code starts


def get_index(a,b):
    t = len(b)
    if a=='Cl':
        return 7
    if a=='CL':
        return 7
    if a=='Br':
        return 8
    if a=='BR':
        return 8
    
    for i in range(t):
        if a[0]==b[i]:
            return i
    return -1



def pocket_coordinate_data_to_file(start,end):
    #########################################################################
    '''
    this function extract the atom coordinates for each atom-pair of protein-ligand complex.
    output is a coordinate file and a description file, the description file records the number of 
    atoms for protein and ligand.
    '''
    #########################################################################
    t1 = len(all_data)
    for i in range(start,end):
        print('process {0}-th '.format(i))
        
        protein = {}
        for ii in range(5):
            protein[Protein_Atom[ii]] = []
            
        name = all_data[i]
        t1 = pre + 'pqr/' + name + '_pocket.pqr'
        f1 = open(t1,'r')
        for line in f1.readlines():
            if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
                atom = line[12:14]
                atom = atom.strip()
                index = get_index(atom,Protein_Atom)
                if index==-1:
                    continue
                else:
                    protein[Protein_Atom[index]].append(line[30:54+8])
        f1.close()
        
        
        ligand = {}
        for ii in range(10):
            ligand[Ligand_Atom[ii]] = []
            
        t2 = pre + 'refined/' + name + '/' + name + '_ligand.mol2'
        f2 = open(t2,'r')
        contents = f2.readlines()
        t3 = len(contents)
        start = 0
        end = 0
        for jj in range(t3):
            if contents[jj][0:13]=='@<TRIPOS>ATOM':
                start = jj + 1
                continue
            if contents[jj][0:13]=='@<TRIPOS>BOND':
                end = jj - 1
                break
        for kk in range(start,end+1):
            if contents[kk][8:17]=='thiophene':
                print('thiophene',kk)
            atom = contents[kk][47:49]
            atom = atom.strip()
            index = get_index(atom,Ligand_Atom)
            if index==-1:
                continue
            else:
                temp = [ contents[kk][17:46] , contents[kk][70:76] ]
                ligand[Ligand_Atom[index]].append(temp)
        f2.close()
        
        
        for i in range(5):
            for j in range(10):
                l_atom = ligand[ Ligand_Atom[j] ]
                p_atom = protein[ Protein_Atom[i] ]
                number_p = len(p_atom)
                number_l = len(l_atom)
                number_all = number_p + number_l
        
                all_atom = np.zeros((number_all,5))
                for jj in range(number_p):
                    all_atom[jj][0] = float(p_atom[jj][0:8])
                    all_atom[jj][1] = float(p_atom[jj][8:16])
                    all_atom[jj][2] = float(p_atom[jj][16:24])
                    all_atom[jj][3] = 1
                    all_atom[jj][4] = float(p_atom[jj][24:32])
                for jjj in range(number_p,number_all):
                    all_atom[jjj][0] = float(l_atom[jjj-number_p][0][0:9])
                    all_atom[jjj][1] = float(l_atom[jjj-number_p][0][9:19])
                    all_atom[jjj][2] = float(l_atom[jjj-number_p][0][19:29])
                    all_atom[jjj][3] = 2
                    all_atom[jjj][4] = float(l_atom[jjj-number_p][1])
        
                filename2 = pre + 'pocket_coordinate_with_charge/' + name + '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_coordinate.csv'
                np.savetxt(filename2,all_atom,delimiter=',')
                filename3 = pre + 'pocket_coordinate_with_charge/' + name +  '_' + Protein_Atom[i] + '_' + Ligand_Atom[j] + '_protein_ligand_number.csv'
                temp = np.array(([number_p,number_l]))
                np.savetxt(filename3,temp,delimiter=',')
        
#############################################################################################   
# extract coordinate code ends




############################################################################################
# create dowker complex start
def distance_of_two_points(p1,p2):
    temp = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
    res = pow(temp,0.5)
    return res


def charge_distance_of_two_points(p1,p2):
    dis = distance_of_two_points(p1, p2)
    temp1 = 100 * p1[4]*p2[4]/dis
    temp2 = 1 + math.exp(-temp1)
    res = 1.0/temp2
    return res
    
    
def get_protein_index_in_P(p,P):
    for i in range(len(P)):
        if p==P[i]:
            return i


def get_ligand_index_in_L(l,L):
    for i in range(len(L)):
        if l==L[i]:
            return i


def get_dowker_complex(N,cutoff,filtration,PI,LI):
    #####################################################################################
    # read pocket coordinate
    
    name = all_data[N]
    filename = pre + 'pocket_coordinate_with_charge/' + name + '_' + Protein_Atom[PI] + '_' + Ligand_Atom[LI] + '_coordinate.csv'
    point_cloud = np.loadtxt(filename,delimiter=',')
    filename = pre + 'pocket_coordinate_with_charge/' + name + '_' + Protein_Atom[PI] + '_' + Ligand_Atom[LI] + '_protein_ligand_number.csv'
    temp = np.loadtxt(filename,delimiter=',')

    p_number = int(temp[0])
    l_number = int(temp[1])
    
    ######################################################################################
    # use cutoff distance to extract the binding core atoms
    P = []
    L = []
    # protein atoms
    for i in range(p_number):
        for j in range(p_number,p_number+l_number):
            dis = distance_of_two_points(point_cloud[i],point_cloud[j])
            if dis<=cutoff:
                P.append(i)
                break
    # ligand atoms
    for i in range(p_number,p_number+l_number):
        L.append(i)
    
    ####################################################################################
    #  create the distance list between protein and ligand, then sort
    dis_list = []
    for i in range(len(P)):
        for j in range(len(L)):
            p = P[i]
            l = L[j]
            dis = distance_of_two_points(point_cloud[p], point_cloud[l])
            if dis<=filtration:
                c_dis = charge_distance_of_two_points(point_cloud[p], point_cloud[l])
                dis_list.append([ p,l,c_dis ])
    dis_list = sorted(dis_list,key=lambda x:(x[2]))
    
    ###################################################################################
    # create filtered dowker complex, the component in protein
    
    simplices_p = []
    count_p = 0
    L_neighbour = []
    is_edge_matrix = np.zeros((p_number,p_number))
    #is_triangle_matrix = np.zeros((p_number,p_number,p_number))
    
    
    for i in range(len(L)):
        L_neighbour.append([])
        
    for i in range(len(P)):
        temp = [ count_p, 0, 0, P[i] ]
        simplices_p.append(temp)
        count_p = count_p + 1
    
    for i in range(len(dis_list)):
        p = dis_list[i][0]
        l = dis_list[i][1]
        filtration = dis_list[i][2]
        p_index = get_protein_index_in_P(p,P)   # relative index
        l_index = get_ligand_index_in_L(l,L)    # relative index
        number_had = len(L_neighbour[l_index])
        
        if number_had==0:
            L_neighbour[l_index].append(p)
        elif number_had==1:
            L_neighbour[l_index].append(p)
            one = min( L_neighbour[l_index][0], p )
            two = max( L_neighbour[l_index][0], p )
            if is_edge_matrix[one][two]==0:
                is_edge_matrix[one][two] = 1
                temp = [ count_p, filtration, 1, one, two ]
                simplices_p.append(temp)
                count_p = count_p + 1
        else:
            t = len(L_neighbour[l_index])
            for tt in range(t):
                one = L_neighbour[l_index][tt]
                m = min(one,p)
                M = max(one,p)
                if is_edge_matrix[m][M]==0:
                    is_edge_matrix[m][M] = 1
                    temp = [ count_p, filtration, 1, m, M ]
                    simplices_p.append(temp)
                    count_p = count_p + 1
            
            #for ii in range(t):
            #    one = L_neighbour[l_index][ii]
            #    for jj in range(ii+1,t):
            #        two = L_neighbour[l_index][jj]
            #        m = min(one,two,p)
            #        M = max(one,two,p)
            #        mid = one + two + p - m - M
            #        if is_triangle_matrix[m][mid][M]==0:
            #            is_triangle_matrix[m][mid][M] = 1
            #            temp = [ count_p, filtration, 2, m, mid, M ]
            #            simplices_p.append(temp)
            #            count_p = count_p + 1
            
            L_neighbour[l_index].append(p)
    
    ##################################################################################
    # create filtered dowker complex, the component in ligand
    
    simplices_l = []
    count_l = 0
    P_neighbour = []
    edges = []
    #triangles = []
    
    for i in range(len(P)):
        P_neighbour.append([])
    for i in range(len(L)):
        temp = [ count_l, 0, 0, L[i] ]
        simplices_l.append(temp)
        count_l = count_l + 1
    
    for i in range(len(dis_list)):
        p = dis_list[i][0]
        l = dis_list[i][1]
        filtration = dis_list[i][2]
        p_index = get_protein_index_in_P(p,P)   # relative index
        l_index = get_ligand_index_in_L(l,L)    # relative index
        number_had = len(P_neighbour[p_index])
        
        if number_had==0:
            P_neighbour[p_index].append(l)
        elif number_had==1:
            P_neighbour[p_index].append(l)
            one = min( P_neighbour[p_index][0], l )
            two = max( P_neighbour[p_index][0], l )
            if ([one,two] in edges)==False:
                edges.append([one,two])    
                temp = [ count_l, filtration, 1, one, two ]
                simplices_l.append(temp)
                count_l = count_l + 1
        else:
            t = len(P_neighbour[p_index])
            for tt in range(t):
                one = P_neighbour[p_index][tt]
                m = min(one,l)
                M = max(one,l)
                if ([m,M] in edges)==False:
                    edges.append([m,M])
                    temp = [ count_l, filtration, 1, m, M ]
                    simplices_l.append(temp)
                    count_l = count_l + 1
            
            #for ii in range(t):
            #    one = P_neighbour[p_index][ii]
            #    for jj in range(ii+1,t):
            #        two = P_neighbour[p_index][jj]
            #        m = min(one,two,l)
            #        M = max(one,two,l)
            #        mid = one + two + l - m - M
            #        if ([m,mid,M] in triangles)==False:
            #            triangles.append([m,mid,M])
            #            temp = [ count_l, filtration, 2, m, mid, M ]
            #            simplices_l.append(temp)
            #            count_l = count_l + 1
            P_neighbour[p_index].append(l)
    
    return simplices_p,simplices_l



###########################################################################################
# get persistent spectral information start
def get_point_index(point,points):
    for i in range(len(points)):
        if point==points[i]:
            return i
   
    
def get_edge_index(p1,p2,edges):
    for i in range(len(edges)):
        if (p1==edges[i][0])&(p2==edges[i][1]):
            return i
    
    
def eigenvalue0_of_each_combination_to_file(typ,simplices,name,P,L,cutoff,filtration,grid):
    #print('process {0}-{1} combination of {2}'.format(P,L,name))
    pre1 = pre + 'charge_eigenvalue_' + str(cutoff) + '_' + str(filtration) + '_'  + 'zero/'
    
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre1 + name + '_' + P + '_' + L + '_' + typ + '_eigenvalue_0D.txt'
        res = [-1]
        f = open(filename1,'w')
        f.writelines(str(res))
        f.close()
        
        return
        
    
    #get 0-dimension laplacian
    
    number0 = int(1/grid)
    
    eigenvalue0 = [1] # have complex, use 1 in the first position as a signal
    for i in range(number0):
        filtra0 = i * grid
        points = []
        edges = []
        for r in range(len(simplices)):
            if simplices[r][1]<=filtra0:
                if simplices[r][2]==0:
                    points.append(simplices[r][3])
                elif simplices[r][2]==1:
                    edges.append([ simplices[r][3] , simplices[r][4] ])
                
            else:
                break
        
        
        row = len(points)
        column = len(edges)
        
        if column==0:
            # only have points, no edges
            res = []
            for ii in range(row):
                res.append(0)
            eigenvalue0.append(res)
            
        else:
            zero_boundary = np.zeros((row,column))
            for j in range(column):
                one = edges[j][0]
                two = edges[j][1]
                index1 = get_point_index(one, points)
                index2 = get_point_index(two, points)
                zero_boundary[index1][j] = -1
                zero_boundary[index2][j] = 1
            Laplacian = np.dot( zero_boundary, zero_boundary.T )
            values = np.linalg.eigvalsh(Laplacian)
            res = []
            for iii in range(len(values)):
                res.append( values[iii] )
            eigenvalue0.append(res)
    
    
    filename1 = pre1 + name + '_' + P + '_' + L + '_' + typ + '_eigenvalue_0D.txt'
    f = open(filename1,'w')
    f.writelines(str(eigenvalue0))
    f.close()
    

def eigenvalue_to_file(start,end,cutoff,filtration,grid):
    for i in range(start,end):
        print(i)
        name = all_data[i]
        #print('process {0}-data, {1}'.format(i,name))
        for P in range(5):
            for L in range(10):
                simplices_p,simplices_l = get_dowker_complex(i,cutoff,filtration,P,L)
                eigenvalue0_of_each_combination_to_file('protein',simplices_p,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
                eigenvalue0_of_each_combination_to_file('ligand',simplices_l,name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
                

#########################################################################################
# get persistent spectral information ends             
                



#########################################################################################
# feature generation starts

    
def get_spectral_moment(ls,k):
    if len(ls)==0:
        return 0
    res = 0
    for i in range(len(ls)):
        if ls[i]!=0:
            res = res + pow(ls[i],k)
    return res



def test_feature_to_file(typ,start,end,cutoff,grid):
    row = end - start
    N = 11
    number0 = 50
    number1 = 50
    add_value = int(100*grid)
    add_value = 1
    c0 = int(1/grid)
    c1 = int(1/grid)
    column = 50 * c0 * N + 50 * c1 * N
    
    feature_matrix = np.zeros((row,column))
    pre1 = pre + 'charge_eigenvalue_10_10_zero/' 
    
    
    for i in range(start,end):
        print(i)
        name = test_data[i]
        count = 0
        for P in range(5):
            for L in range(10):
                filename0 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'protein_eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'ligand_eigenvalue_0D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                
                
                if eigenvalue0[0]==-1:
                    for ii in range(0,number0,add_value):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                    #for ii in range(number1):
                    #    for iii in range(N):
                    #        feature_matrix[i-start][count] = 0
                    #        count = count + 1
                else:
                    #number0 = 2
                    for ii in range(1,number0+1,add_value):
                        value = []
                        all_value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                                all_value.append(0)
                            else:
                                value.append(v)
                                all_value.append(v)
                            #print(value)
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-5)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-2)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,0)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,2)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,5)
                        count = count + 1
                        
                ##########################################################################################
                        
                if eigenvalue1[0]==-1:
                    for ii in range(0,number1,add_value):
                        for jj in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                else:
                    for ii in range(1,number1+1,add_value):
                        value = []
                        all_value = []
                        c0 = 0
                        for iii in range(len(eigenvalue1[ii])):
                            v = eigenvalue1[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                                all_value.append(0)
                            else:
                                value.append(v)
                                all_value.append(v)
                            #print(value)
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-5)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-2)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,0)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,2)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,5)
                        count = count + 1
                           
                 ###########################################################################################
                    
                        
                        
    filename = pre + 'pocket_feature/charge_test.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')





def train_feature_to_file(typ,start,end,cutoff,grid):
    row = end - start
    N = 11
    number0 = 50
    number1 = 50
    add_value = int(100*grid)
    add_value = 1
    c0 = int(1/grid)
    c1 = int(1/grid)
    column = 50 * c0 * N + 50 * c1 * N
    
    feature_matrix = np.zeros((row,column))
    pre1 = pre + 'charge_eigenvalue_10_10_zero/' 
    
    
    for i in range(start,end):
        print(i)
        name = train_data[i]
        count = 0
        for P in range(5):
            for L in range(10):
                filename0 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'protein_eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'ligand_eigenvalue_0D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                
                
                if eigenvalue0[0]==-1:
                    for ii in range(0,number0,add_value):
                        for iii in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                    #for ii in range(number1):
                    #    for iii in range(N):
                    #        feature_matrix[i-start][count] = 0
                    #        count = count + 1
                else:
                    #number0 = 2
                    for ii in range(1,number0+1,add_value):
                        value = []
                        all_value = []
                        c0 = 0
                        for iii in range(len(eigenvalue0[ii])):
                            v = eigenvalue0[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                                all_value.append(0)
                            else:
                                value.append(v)
                                all_value.append(v)
                            #print(value)
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-5)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-2)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,0)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,2)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,5)
                        count = count + 1
                        
                ##########################################################################################
                        
                if eigenvalue1[0]==-1:
                    for ii in range(0,number1,add_value):
                        for jj in range(N):
                            feature_matrix[i-start][count] = 0
                            count = count + 1
                else:
                    for ii in range(1,number1+1,add_value):
                        value = []
                        all_value = []
                        c0 = 0
                        for iii in range(len(eigenvalue1[ii])):
                            v = eigenvalue1[ii][iii]
                            if v<=0.000000001:
                                c0 = c0 + 1
                                all_value.append(0)
                            else:
                                value.append(v)
                                all_value.append(v)
                            #print(value)
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-5)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-2)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,-1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,0)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,1)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,2)
                        count = count + 1
                        
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,3)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,4)
                        count = count + 1
                           
                        feature_matrix[i-start][count] = get_spectral_moment(all_value,5)
                        count = count + 1
                           
                 ###########################################################################################
                    
                        
                        
    filename = pre + 'pocket_feature/charge_train.csv'
    np.savetxt(filename,feature_matrix,delimiter=',')



    
def get_name_index(name,contents):
    t = len(contents)
    for i in range(t):
        if contents[i][0:4]==name:
            return i


def get_target_matrix_of_train():
    t = len(train_data)
    target_matrix = []
    t1 = pre + Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = train_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + 'pocket_feature/' + 'target_matrix_of_train.csv',res,delimiter=',')


def get_target_matrix_of_test():
    t = len(test_data)
    target_matrix = []
    t1 = pre +  Year + '_INDEX_refined.data'
    f1 = open(t1,'r')
    contents = f1.readlines()
    f1.close()
    for i in range(t):  # tttttttttttttttttttttttttttttttttt
        name = test_data[i]
        index = get_name_index(name,contents)
        target_matrix.append(float(contents[index][18:23]))
    res = np.array(target_matrix)
    np.savetxt(pre + 'pocket_feature/' + 'target_matrix_of_test.csv',res,delimiter=',')
    


# feature generation code ends.
###########################################################################################################



############################################################################################################
# machine_learning algorithm starts.
    
def gradient_boosting(X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 40000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse1 = mean_squared_error(Y_test, regr.predict(X_test))
    mse2 = pow(mse1,0.5)
    #mse3 = mse2/0.7335
    mse3 = mse2
    return [pearson_coorelation[0],mse3]


def get_pearson_correlation(typ,pref):
    feature_matrix_of_train = np.loadtxt( pre + 'pocket_feature/' + 'charge_train.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( pre + 'pocket_feature/' + 'target_matrix_of_train.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( pre + 'pocket_feature/' + 'charge_test.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( pre + 'pocket_feature/' +  'target_matrix_of_test.csv',delimiter=',' )
    
    
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1)) 
    print(feature_matrix_of_train.shape)
    for i in range(number):
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test)
        print(P[i])
    median_p = np.median(P)
    median_m = np.median(M)
    print('for data ' + Year + ', 10 results for ' + typ + '-model are:')
    print(P)
    print('median pearson correlation values are')
    print(median_p)
    print('median mean squared error values are')
    print(median_m)
    
    
############################################################################################################
# machine_learning algorithm ends.


def run_for_PDBbind_2013():
    ##############################################################
    '''
    by running this function, you can get the results for data2013
    '''
    ##############################################################
    
    
    # extract coordinate
    pocket_coordinate_data_to_file(0,2959) 
    
    # create dowker complex and compute the spectral information
    eigenvalue_to_file(0,2959,10,10,0.02)
    
    # feature generation
    train_feature_to_file(0,2764,10,0.02)
    test_feature_to_file(0,195,10,0.02)
    get_target_matrix_of_train()
    get_target_matrix_of_test()
    
    
    # machine learning
    get_pearson_correlation('charge','')
    
    
run_for_PDBbind_2013()


