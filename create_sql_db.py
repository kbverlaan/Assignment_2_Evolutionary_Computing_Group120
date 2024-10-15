import sqlite3
from sqlite3 import Error
# from create_sql_db import *
# 2 alg, 1st ea2 and 2nd evol framework, 2 enemy groups (tbd)
# for each combo, we do 10 runs
# for every run mean and max fitness
# output for every gen island and mean done in evolve

def create_connection(db_file):
    # create a database connection to a SQLite database
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("Database successfully created")
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def create_table_1(cursor):
    cursor.execute('''DROP TABLE IF EXIST Alg_ea2;
                    CREATE TABLE Alg_ea2,
                    (id INTEGER PRIMARY KEY,
                    enemy_group INT,
                    run INT,
                    generation INT,
                    population INT,
                    fitness DECIMAL(10,4));'''
                   )

def create_table_2(cursor):
    cursor.execute('''DROP TABLE IF EXIST Alg_evol_framework;
                    CREATE TABLE Alg_evol_framework,
                    (id INTEGER PRIMARY KEY,
                    enemy_group INT,
                    run INT,
                    generation INT,
                    population INT,
                    fitness DECIMAL(10,4));'''
                   )
def save_output(num_to_exchange, island_name, target_name):
#     save_output((enemy_group, run, generation, population, fitness))
    cursor.execute("INSERT INTO Alg_ea2 (enemy_group, run, generation, population, fitness) VALUES (?,?,?,?,?)",
                                (enemy_group, run, generation, population, fitness))

def create_and_init_db():
    create_connection(r"db_file.db")
    conn = sqlite3.connect('db_file.db')
    cursor = conn.cursor()
    create_table_1()
    create_table_2()
    save_output()

