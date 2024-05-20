import csv
import random

listOfStudents = []

with open('data.csv', encoding='utf-8') as f:
    reader = csv.reader(f)

    for row in reader:
        listOfStudents.append(row)

listOfStudentsWithoutHeaders = listOfStudents[1:]

lenOfListOfStudentsWithoutHeadersBeforeAddingNewData = len(listOfStudentsWithoutHeaders)

ukupno_godina_do_diplomiranja = [inner_list[1] for inner_list in listOfStudentsWithoutHeaders]
derivated_nacin_studiranja_beautified = [inner_list[2] for inner_list in listOfStudentsWithoutHeaders]
derivated_srednja_skola = [inner_list[3] for inner_list in listOfStudentsWithoutHeaders]
opstina_rodjenja = [inner_list[4] for inner_list in listOfStudentsWithoutHeaders]
opstina_prebivalista = [inner_list[5] for inner_list in listOfStudentsWithoutHeaders]
spol = [inner_list[6] for inner_list in listOfStudentsWithoutHeaders]

for i in range(100000):
    newRandomRow = []

    newRandomRow.append('_' + str(i))
    newRandomRow.append(ukupno_godina_do_diplomiranja[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])
    newRandomRow.append(derivated_nacin_studiranja_beautified[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])
    newRandomRow.append(derivated_srednja_skola[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])
    newRandomRow.append(opstina_rodjenja[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])
    newRandomRow.append(opstina_prebivalista[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])
    newRandomRow.append(spol[random.randint(0, lenOfListOfStudentsWithoutHeadersBeforeAddingNewData - 1)])

    listOfStudentsWithoutHeaders.append(newRandomRow)

oldStudentsAndNewRandomStudentsWithHeaders = listOfStudents[:1] + listOfStudentsWithoutHeaders

with open('data_random_1.csv', encoding='utf-8', newline='', mode='w') as f:
    writer = csv.writer(f)

    for row in oldStudentsAndNewRandomStudentsWithHeaders:
        writer.writerow(row)

