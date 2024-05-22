import csv

# Сохранение вакансий в csv file 
def save_to_csv(jobs):
  file = open('test.csv', mode='w')
  writer = csv.writer(file)
  writer.writerow(['Название вакансии', 'Компания', 'Город', 'Ссылка'])
  # Получение значений из словаря 
  for job in jobs:
    writer.writerow(list(job.values()))
  return