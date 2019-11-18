# Time    : 2019/11/6 22:20
# Author  : yangfan
# Email   : thePrestige_yf@outlook.com
# Software: PyCharm

import pymysql
import uuid
import os
import re


def readLine():
    # db = pymysql.connect(host='localhost', user='root', password='root', db='nlpcc2018_7', charset='utf8')
    # with db.cursor() as cursor:
    #     with open("./nlpcc2018.trainset/knowledge/nlpcc-iccpol-2016.kbqa.kb", "r", encoding="UTF-8") as kg:
    #         qgLines = kg.readlines()
    #         counter = 0
    #         for i in range(len(qgLines)):
    #             if (i/(len(qgLines)))>counter/100:
    #                 print("processing: %d%%"%(counter))
    #                 counter += 1
    #             if(qgLines[i]!=""):
    #                 # try:
    #                 t0, t1, t2 = qgLines[i].rstrip("\n").split("|||")
    #                 t0 = pymysql.escape_string(t0.rstrip(" ").lstrip(" "))
    #                 sql_search = "select subject_id from subject_table where subject_name=\'%s\';"%(t0)
    #                 if cursor.execute(sql_search)==0:
    #                     t0_uuid = str(uuid.uuid1())
    #                     sql_sub = "insert into subject_table (subject_id, subject_name) values (\'%s\', \'%s\');" % (t0_uuid, t0)
    #                     cursor.execute(sql_sub)
    #                     db.commit()
    #                 else:
    #                     t0_uuid, =cursor.fetchone()
    #                 t1 = pymysql.escape_string(t1.rstrip(" ").lstrip(" ").lstrip("- ").lstrip("• "))
    #                 sql_search = "select relation_id from relation_table where relation_name=\'%s\';"%(t1)
    #                 if cursor.execute(sql_search)==0:
    #                     t1_uuid = str(uuid.uuid1())
    #                     sql_rel = "insert into relation_table (relation_id, relation_name) values (\'%s\', \'%s\');" % (t1_uuid, t1)
    #                     cursor.execute(sql_rel)
    #                     db.commit()
    #                 else:
    #                     t1_uuid, =cursor.fetchone()
    #                 t2 = pymysql.escape_string(t2.rstrip(" ").lstrip(" ").rstrip("\\"))
    #                 sql_search = "select object_id from object_table where object_name=\'%s\'"%(t2)
    #                 if cursor.execute(sql_search)==0:
    #                     t2_uuid = str(uuid.uuid1())
    #                     sql_obj = "insert into object_table (object_id, object_name) values (\'%s\', \'%s\');" % (t2_uuid, t2)
    #                     cursor.execute(sql_obj)
    #                     db.commit()
    #                 else:
    #                     t2_uuid, = cursor.fetchone()
    #                 sql_search = "select * from triple_table where subject_id=\'%s\' and relation_id=\'%s\' and object_id=\'%s\';"%(t0_uuid, t1_uuid, t2_uuid)
    #                 if cursor.execute(sql_search)==0:
    #                     triple_uuid=str(uuid.uuid1())
    #                     sql_triple = "insert into triple_table (triple_id, subject_id, relation_id, object_id) values (\'%s\',\'%s\',\'%s\',\'%s\');"%(triple_uuid, t0_uuid, t1_uuid, t2_uuid)
    #                     cursor.execute(sql_triple)
    #                     db.commit()
    #                 # except:
    #                 #     print("error at : "+str(i))
    #print("=====================finished insert kg into mysql=====================")
    with open("./nlpcc2018.trainset/nlpcc2018.kbqa.train", "r", encoding="UTF-8") as qa:
        with open("./nlpcc2018.trainset/nlpcc2018.kbqg.train", "r", encoding="UTF-8") as qg:
            with open("./nlpcc2018.trainset/ner.txt", "w", encoding="UTF-8") as out:
                aLines = qa.readlines()
                gLines = qg.readlines()
                for i in range(0,len(aLines),3):
                    problem = re.sub("[^0-9A-Za-z\u4e00-\u9fa5]", "", aLines[i].split("\t")[1].rstrip("\n"))
                    triple = gLines[i].split("\t")[1]
                    sub, rel, obj = triple.split("|||")
                    sub = sub.rstrip(" ").lstrip(" ")
                    index = problem.find(sub)
                    if(index!=-1):
                        bio = "O"*index+"B"+"I"*(len(sub)-1)+"O"*(len(problem)-len(sub)-index)
                        # print(problem)
                        # print(bio)
                        out.write(problem+"\n")
                        out.write(bio+"\n")
                    #rel = sub.rstrip(" ").lstrip(" ")
                    #line = "\t".join([problem, sub, rel])
                    #line += '\n'
                    #out.write(line)



if __name__=='__main__':
    readLine()