# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import scipy as sp
import os,sys

Protein_Atom = ['C','N','O','S']
Ligand_Atom = ['C','N','O','S','P','F','Cl','Br','I']
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']



pre = './'

f1 = open(pre + '2013_screening_name.txt')
pre_test_data = f1.readlines()
test_data = eval(pre_test_data[0])
f1.close()

f = open('./positive_train_index.txt')
pre_index = f.read()
positive_index = eval(pre_index)
f.close()

########################################################################################
# extract coordinate code starts
def get_index(a,b):
    t = len(b)
    if a=='Cl':
        return 6
    if a=='CL':
        return 6
    if a=='Br':
        return 7
    if a=='BR':
        return 7
    
    for i in range(t):
        if a[0]==b[i]:
            return i
    return -1

def test_coordinate_data_to_file(start,end):
    t1 = len(test_data)
    for i in range(start,end):
        #print('process {0}-th '.format(i))
        
        protein = {}
        for ii in range(4):
            protein[Protein_Atom[ii]] = []
            
        name = test_data[i]
        print(i,name)
        t1 = pre + 'refined/' + name + '/' + name + '_pocket.pdb'
        f1 = open(t1,'r')
        for line in f1.readlines():
            if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
                atom = line[12:14]
                atom = atom.strip()
                index = get_index(atom,Protein_Atom)
                if index==-1:
                    continue
                else:
                    protein[Protein_Atom[index]].append(line[30:54])
        f1.close()
        
        
        # test ligand
        t2 = pre + 'screening_power/poses/' + name + '_testing_poses.sdf'
        f2 = open(t2,'r')
        contents = f2.readlines()
        f2.close()
        train_index_list = [0]
        for ii in range(len(contents)):
            if contents[ii][0:4]=='$$$$':
                train_index_list.append(ii)
        for temp_i in range(len(train_index_list)-1):
            top = train_index_list[temp_i]
            bottom = train_index_list[temp_i+1]
            
            ligand = {}
            for ii in range(9):
                ligand[Ligand_Atom[ii]] = []
            pose_name = ''
            start = 0
            for jj in range(top,bottom):
                if contents[jj][0:4]==name:
                    pose_name = contents[jj][5:9]
                    start = jj + 4
                    break
            
            end = 0
            start_len = len( contents[start].split() )
            for jj in range(start,bottom):
                t = len( contents[jj].split() )
                if t== start_len:
                    end = jj
                else:
                    break
            
            for kk in range(start,end+1):
                temp = contents[kk].split()
                atom = temp[3].strip()
                index = get_index(atom,Ligand_Atom)
                if index==-1:
                    continue
                else:
                    ligand[Ligand_Atom[index]].append([ float(temp[0]),float(temp[1]),float(temp[2])  ])
            
        
        
            for pi in range(4):
                for lj in range(9):
                    l_atom = ligand[ Ligand_Atom[lj] ]
                    p_atom = protein[ Protein_Atom[pi] ]
                    number_p = len(p_atom)
                    number_l = len(l_atom)
                    number_all = number_p + number_l
            
                    all_atom = np.zeros((number_all,4))
                    for jj in range(number_p):
                        all_atom[jj][0] = float(p_atom[jj][0:8])
                        all_atom[jj][1] = float(p_atom[jj][8:16])
                        all_atom[jj][2] = float(p_atom[jj][16:24])
                        all_atom[jj][3] = 1
                    for jjj in range(number_p,number_all):
                        all_atom[jjj][0] = l_atom[jjj-number_p][0]
                        all_atom[jjj][1] = l_atom[jjj-number_p][1]
                        all_atom[jjj][2] = l_atom[jjj-number_p][2]
                        all_atom[jjj][3] = 2
            
                    filename2 = pre + 'point_10/' + name + '/test/' + pose_name + '_' + Protein_Atom[pi] + '_' + Ligand_Atom[lj] + '_coordinate.npy'
                    np.save(filename2,all_atom)
                    filename3 = pre + 'point_10/' + name +  '/test/' + pose_name + '_' + Protein_Atom[pi] + '_' + Ligand_Atom[lj] + '_number.npy'
                    temp = np.array(([number_p,number_l]))
                    np.save(filename3,temp)
        
    
           


def train_coordinate_data_to_file(start,end):
    t1 = len(test_data)
    for i in range(start,end):
        #print('process {0}-th '.format(i))
        
        protein = {}
        for ii in range(4):
            protein[Protein_Atom[ii]] = []
            
        name = test_data[i]
        print(i,name)
        t1 = pre + 'refined/' + name + '/' + name + '_pocket.pdb'
        f1 = open(t1,'r')
        for line in f1.readlines():
            if (line[0:4]=='ATOM')&(line[17:20] in aa_list ):
                atom = line[12:14]
                atom = atom.strip()
                index = get_index(atom,Protein_Atom)
                if index==-1:
                    continue
                else:
                    protein[Protein_Atom[index]].append(line[30:54])
        f1.close()
        
        
        # train ligand
        t2 = pre + 'screening_power/poses/' + name + '_training_poses.sdf'
        f2 = open(t2,'r')
        contents = f2.readlines()
        f2.close()
        train_index_list = [0]
        for ii in range(len(contents)):
            if contents[ii][0:4]=='$$$$':
                train_index_list.append(ii)
        #print('coordinate train',len(train_index_list))
        
        for temp_i in range(len(train_index_list)-1):
            top = train_index_list[temp_i]
            bottom = train_index_list[temp_i+1]
            
            ligand = {}
            for ii in range(9):
                ligand[Ligand_Atom[ii]] = []
            pose_name = ''
            start = 0
            for jj in range(top,bottom):
                if contents[jj][0:4]==name:
                    pose_name = contents[jj][5:9]
                    start = jj + 4
                    break
            
            end = 0
            start_len = len( contents[start].split() )
            for jj in range(start,bottom):
                t = len( contents[jj].split() )
                if t== start_len:
                    end = jj
                else:
                    break
            
            for kk in range(start,end+1):
                temp = contents[kk].split()
                atom = temp[3].strip()
                index = get_index(atom,Ligand_Atom)
                if index==-1:
                    continue
                else:
                    ligand[Ligand_Atom[index]].append([ float(temp[0]),float(temp[1]),float(temp[2])  ])
            
        
        
            for pi in range(4):
                for lj in range(9):
                    l_atom = ligand[ Ligand_Atom[lj] ]
                    p_atom = protein[ Protein_Atom[pi] ]
                    number_p = len(p_atom)
                    number_l = len(l_atom)
                    number_all = number_p + number_l
            
                    all_atom = np.zeros((number_all,4))
                    for jj in range(number_p):
                        all_atom[jj][0] = float(p_atom[jj][0:8])
                        all_atom[jj][1] = float(p_atom[jj][8:16])
                        all_atom[jj][2] = float(p_atom[jj][16:24])
                        all_atom[jj][3] = 1
                    for jjj in range(number_p,number_all):
                        all_atom[jjj][0] = l_atom[jjj-number_p][0]
                        all_atom[jjj][1] = l_atom[jjj-number_p][1]
                        all_atom[jjj][2] = l_atom[jjj-number_p][2]
                        all_atom[jjj][3] = 2
            
                    filename2 = pre + 'point_10/' + name + '/train/' + pose_name + '_' + Protein_Atom[pi] + '_' + Ligand_Atom[lj] + '_coordinate.npy'
                    np.save(filename2,all_atom)
                    filename3 = pre + 'point_10/' + name +  '/train/' + pose_name + '_' + Protein_Atom[pi] + '_' + Ligand_Atom[lj] + '_number.npy'
                    temp = np.array(([number_p,number_l]))
                    np.save(filename3,temp)
        
        

############################################################################################
# create dowker complex start
def distance_of_two_points(p1,p2):
    temp = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
    res = pow(temp,0.5)
    return res
    

def get_protein_index_in_P(p,P):
    for i in range(len(P)):
        if p==P[i]:
            return i


def get_ligand_index_in_L(l,L):
    for i in range(len(L)):
        if l==L[i]:
            return i


def get_dowker_complex(typ,N,pose_name,cutoff,filtration,PI,LI):
    # read pocket coordinate
    name = test_data[N]
    filename = pre + 'point_10/' + name + '/' + typ + '/' + pose_name + '_' + Protein_Atom[PI] + '_' + Ligand_Atom[LI] + '_coordinate.npy'
    point_cloud = np.load(filename)
    filename = pre + 'point_10/' + name + '/' + typ + '/' + pose_name + '_' + Protein_Atom[PI] + '_' + Ligand_Atom[LI] + '_number.npy'
    temp = np.load(filename)
    p_number = int(temp[0])
    l_number = int(temp[1])
    if p_number==0 or l_number==0:
        return [],[]
        
    
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
                dis_list.append([ p,l,dis ])
    dis_list = sorted(dis_list,key=lambda x:(x[2]))
    
    ###################################################################################
    # create filtered dowker complex, the component in protein
    
    simplices_p = []
    count_p = 0
    L_neighbour = []
    is_edge_matrix = np.zeros((p_number,p_number))
    #is_triangle_matrix
    
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
    
    
def eigenvalue0_of_each_combination_to_file(typ1,typ,simplices,name,pose_name,P,L,cutoff,filtration,grid):
    #print('process {0}-{1} combination of {2}'.format(P,L,name))
    pre1 = pre + 'eigenvalue_' + str(cutoff) + '_' + str(filtration) + '_'  + 'zero/' + name + '/' + typ1 + '/'
    
    
    if len(simplices)==0:
        # no complex, use -1 in the first position as a signal 
        filename1 = pre1 + pose_name + '_' + P + '_' + L + '_' + typ + '_eigenvalue_0D.txt'
        res = [-1]
        f = open(filename1,'w')
        f.writelines(str(res))
        f.close()
        return
        
    
    #get 0-dimension laplacian
    number0 = int((filtration-2)/grid)
    eigenvalue0 = [1] # have complex, use 1 in the first position as a signal
    for i in range(number0 + 1):
        # get eigenvalue for each filtra0 value with grid from 2 to filtration
        filtra0 = 2 + i * grid
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
    
    
    filename1 = pre1 + pose_name + '_' + P + '_' + L + '_' + typ + '_eigenvalue_0D.txt'
    f = open(filename1,'w')
    f.writelines(str(eigenvalue0))
    f.close()
    

def test_eigenvalue_to_file(start,end,cutoff,filtration,grid):
    for i in range(start,end):
        name = test_data[i]
        print('process {0}-data, {1}'.format(i,name))
        filename = pre + '/test_name.txt'
        f = open(filename)
        pre_contents = f.read()
        contents = eval(pre_contents)
        f.close()
        for j in range(len(contents)):
            pose_name = contents[j]
            print(name,pose_name)
            
            for P in range(4):
                for L in range(9):
                    simplices_p,simplices_l = get_dowker_complex('test',i,pose_name,cutoff,filtration,P,L)
                    eigenvalue0_of_each_combination_to_file('test','protein',simplices_p,name,pose_name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
                    eigenvalue0_of_each_combination_to_file('test','ligand',simplices_l,name,pose_name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
            
                
                        
#########################################################################################
# get persistent spectral information ends




def train_eigenvalue_to_file(start,end,cutoff,filtration,grid):
    for i in range(start,end):
        name = test_data[i]
        print('process {0}-data, {1}'.format(i,name))
        filename = pre + 'screening_power/INDEX/' + name + '_train.data'
        f = open(filename)
        contents = f.readlines()
        f.close()
        #print('eigenvalue train',len(contents))
        
        for j in range(len(contents)):
            pose_name = contents[j][5:9]
            print(name,pose_name,j,len(contents))
            
            for P in range(4):
                for L in range(9):
                    simplices_p,simplices_l = get_dowker_complex('train',i,pose_name,cutoff,filtration,P,L)
                    eigenvalue0_of_each_combination_to_file('train','protein',simplices_p,name,pose_name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
                    eigenvalue0_of_each_combination_to_file('train','ligand',simplices_l,name,pose_name,Protein_Atom[P],Ligand_Atom[L],cutoff,filtration,grid)
        
 

def create_coordinate_dir(start,end):
    for i in range(start,end):
        name = test_data[i]
        cm3 = 'mkdir ./point_10/' + name
        os.system(cm3)
        cm1 = 'mkdir ./point_10/' + name + '/test'
        cm2 = 'mkdir ./point_10/' + name + '/train'
        os.system(cm1)
        os.system(cm2)
        
def create_eigenvalue_dir(start,end):
    for i in range(start,end):
        name = test_data[i]
        cm3 = 'mkdir ./eigenvalue_10_10_zero/' + name
        os.system(cm3)
        cm1 = 'mkdir ./eigenvalue_10_10_zero/' + name + '/test'
        cm2 = 'mkdir ./eigenvalue_10_10_zero/' + name + '/train'
        os.system(cm1)
        os.system(cm2)

           
def before_feature(start,end):
    for i in range(start,end):
        #print(i)
        print('start generate coordinate')
        create_coordinate_dir(i,i+1)
        test_coordinate_data_to_file(i,i+1)
        print('test coordinate ok')
        #train_coordinate_data_to_file(i,i+1)
        print('train coordinate ok')
        create_eigenvalue_dir(i,i+1)
        print('start generate eigenvalue')
        test_eigenvalue_to_file(i,i+1,10,10,0.1)
        print('test eigenvalue ok')
        train_eigenvalue_to_file(i,i+1,10,10,0.1)
        print('train eigenvalue ok')
        

def get_spectral_moment(ls,k):
    if len(ls)==0:
        return 0
    res = 0
    for i in range(len(ls)):
        if ls[i]!=0:
            res = res + pow(ls[i],k)
    return res


def train_feature_to_file(name,cutoff,filtration,grid):
    filename = pre + 'screening_power/INDEX/' + name + '_train.data'
    f = open(filename)
    contents = f.readlines()
    f.close()
    
    row = len(contents)
    N = 11
    number0 = int ((filtration-2)/0.1 )
    number1 = int ((filtration-2)/0.1)
    column = 36 * number0 * N + 36 * number1 * N
    feature_matrix = np.zeros((row,column))
    pre1 = pre + 'eigenvalue_' + str(cutoff) + '_' + str(filtration) + '_zero/' + name + '/train/'
    add_value = 1
    start = 0
    for i in range(row):
        print(i,row)
        pose_name = contents[i][5:9]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre1 + pose_name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'protein_eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + pose_name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'ligand_eigenvalue_0D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
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
                    for ii in range(number1):
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
                    
                        
                        
    filename = pre + 'feature/' + name + '_dis_train.npy'
    np.save(filename,feature_matrix)


    

def test_feature_to_file(name,cutoff,filtration,grid):
    filename = pre + '/test_name.txt'
    f = open(filename)
    pre_contents = f.read()
    contents = eval(pre_contents)
    f.close()
    start = 0
    row = len(contents)
    N = 11
    number0 = int ((filtration-2)/0.1 )
    number1 = int ((filtration-2)/0.1)
    column = 36 * number0 * N + 36 * number1 * N
    feature_matrix = np.zeros((row,column))
    pre1 = pre + 'eigenvalue_' + str(cutoff) + '_' + str(filtration) + '_zero/' + name + '/test/'
    add_value = 1
    
    start = 0
    for i in range(len(contents)):
        print(i)
        pose_name = contents[i]
        count = 0
        for P in range(4):
            for L in range(9):
                filename0 = pre1 + pose_name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'protein_eigenvalue_0D.txt'
                f0 = open(filename0)
                pre_eigenvalue0 = f0.readlines()
                eigenvalue0 = eval(pre_eigenvalue0[0])
                f0.close()
                
                filename1 = pre1 + pose_name + '_' + Protein_Atom[P] + '_' + Ligand_Atom[L] + '_' + 'ligand_eigenvalue_0D.txt'
                f1 = open(filename1)
                pre_eigenvalue1 = f1.readlines()
                eigenvalue1 = eval(pre_eigenvalue1[0])
                f1.close()
                
                
                if eigenvalue0[0]==-1:
                    for ii in range(number0):
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
                    for ii in range(number1):
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
                    
                        
                        
    filename = pre + 'feature/' + name + '_dis_test.npy'
    np.save(filename,feature_matrix)



def train_label_to_file(name):
    filename = pre + 'screening_power/INDEX/' + name + '_train.data'
    f = open(filename)
    contents = f.readlines()
    f.close()
    temp = []
    for line in contents:
        temp2 = line.split()
        temp.append(float(temp2[1]))
    res = np.array(temp)
    filename = pre + 'feature/' + name + '_train_label.npy'
    np.save(filename,res)



def get_feature(start,end):

    for i in range(start,end):
        name = test_data[i]
        train_feature_to_file(name,10,10,0.1)
        test_feature_to_file(name,10,10,0.1)
        train_label_to_file(name)
        



def get_positive_train(i):
    name = test_data[i]
    filename = './feature/' + name + '_dis_train.npy'
    train1 = np.load(filename)
    t1 = train1.shape
    filename = './feature/' + name + '_train_label.npy'
    label1 = np.load(filename)
    
    index = positive_index[i]
    t2 = len(index)
    
    train2 = np.zeros((t2,t1[1]))
    pre_label2 = []
    count = 0
    for i in range(t1[0]):
        if i in index:
            train2[count,:] = train1[i,:]
            pre_label2.append(label1[i])
            count = count + 1
    label2 = np.array(pre_label2)
    
    filename = 'feature-positive/' + name + '_dis_train.npy'
    np.save(filename,train2)
            
    filename = 'feature-positive/' + name + '_train_label.npy'
    np.save(filename,label2)
    
def move(start,end):
    for i in range(start,end):
        get_positive_train(i)
        name = test_data[i]
        cmd = 'cp ./feature/' + name + '_dis_test.npy ./feature-positive/' + name + '_dis_test.npy'
        os.system(cmd)
        #print(i,name)        


def gradient_boosting(X_train,Y_train,X_test):
    params={'n_estimators': 40000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    a_predict = regr.predict(X_test)
    return a_predict




def predict(start,end):

    filename = pre + '/test_name.txt'
    f = open(filename)
    pre_contents = f.read()
    contents = eval(pre_contents)
    f.close()
   
    for i in range(start,end):
        name = test_data[i]
        #print(i,name)
        
        filename1 = pre + '../feature-positive/' + name + '_dis_train.npy'
        train_feature = np.load(filename1)
        
        filename2 = pre + '../feature-positive/' + name + '_train_label.npy'
        train_label = np.load(filename2)
        
        filename3 = pre + '../feature-positive/' + name + '_dis_test.npy'
        test_feature = np.load(filename3)
        
        final_res1 = 0
        rate1 = 0
        
        for j in range(10):
            #print(j)
            res = gradient_boosting(train_feature,train_label,test_feature)
            pre_sort = []
            for k in range(len(res)):
                temp = round(res[k],2)
                pre_sort.append([temp,k])
            index = sorted(pre_sort,key=lambda x:(x[0]),reverse=True)
            top1_index = 2
            same_number = 0
            for kk in range(len(index)-1):
                if index[kk][0]!=index[kk+1][0]:
                    same_number = same_number + 1
                    if same_number==2:
                        top1_index = kk + 1
                        break
                        
            if same_number<2:
                top1_index = 195
            
            
            filename = 'TargetInfo.dat'
            f = open(filename)
            ref = f.readlines()
            f.close()
            box = []
            F_number = 0
            for k in range(8,len(ref)):
                if ref[k][0:4]==name:
                    temp = ref[k].split()
                    F_number = len(temp) - 1
                    for ii in range(1,len(temp)):
                        box.append(temp[ii])
                        
              
            E_number = 0
            for k in range(top1_index):
                index1 = index[k][1]
                 
                if contents[index1] in box:
                    E_number = E_number + 1
            
            EF = 100*E_number/F_number
            #print('top1:',EF)
            final_res1 = final_res1 + EF
            if E_number>0:
                rate1 = rate1 + 1
            
        print(i,name,'EF:',final_res1/10,'success rate:',rate1)
        
        




