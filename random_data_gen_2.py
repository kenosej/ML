import csv
import random
import copy

listOfStudents = []

with open('data.csv', encoding='utf-8') as f:
    reader = csv.reader(f)

    for row in reader:
        listOfStudents.append(row)

listOfStudentsWithoutHeaders = listOfStudents[1:]

lenOfListOfStudentsWithoutHeadersBeforeAddingNewData = len(listOfStudentsWithoutHeaders)

randomlySelectedIndex = []
randomlySelectedStudentsForDuplication = []

def genUniqueIndex():
    index = random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)

    if index in randomlySelectedIndex:
        return genUniqueIndex()

    return index

for i in range(7):
    index = genUniqueIndex()

    randomlySelectedIndex.append(index)

    student = copy.deepcopy(listOfStudentsWithoutHeaders[index])
    student[0] = '_' + str(student[0])

    randomlySelectedStudentsForDuplication.append(student)

counter = 0

for i in range(7777):
    listOfStudentsWithoutHeaders.append(randomlySelectedStudentsForDuplication[counter])

    counter = counter + 1

    if (counter == len(randomlySelectedStudentsForDuplication)):
        counter = 0


oldStudentsAndNewRandomStudentsWithHeaders = listOfStudents[:1] + listOfStudentsWithoutHeaders

with open('data_random_2.csv', encoding='utf-8', newline='', mode='w') as f:
    writer = csv.writer(f)

    for row in oldStudentsAndNewRandomStudentsWithHeaders:
        writer.writerow(row)
