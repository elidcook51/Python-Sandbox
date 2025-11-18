import numpy as np
import random

datafilepath = "C:/Users/ucg8nb/AMPL/hw4problem6.dat"

numStudents = 200
numRooms = 100
hateLength = 10

students = list(range(1, numStudents+ 1))
rooms = list(range(1, numRooms + 1))

with open(datafilepath, 'w') as f:
    f.write(f'set STUDENTS := 1..{numStudents};\n')
    f.write(f'set ROOMS := 1..{numRooms};\n\n')

    f.write('param roomRanking := \n')
    for s in students:
        for r in rooms:
            score = random.randint(1, 100)
            f.write(f"   [{s},{r}] {score}\n")
    f.write(';\n\n')

    for s in students:
        hated = random.sample([x for x in students if x != s], hateLength)
        f.write(f"set HATELIST[{s}] := {' '.join(str(x) for x in hated)};\n")
