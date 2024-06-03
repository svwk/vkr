import json
from locust import HttpUser, task, between

from params_t import serialized_text, html_text

# Тестовые данные
text2 = {"text": serialized_text}
txt3 = {"text": html_text}
txt4 = json.dumps(txt3)


# class WebsiteTestUser(HttpUser):
#     wait_time = between(0, 0)

# @task(1)
# def root_access(self):
#     self.client.get("/")


class WebsiteTestUser2(HttpUser):
    wait_time = between(0, 0)

    @task(1)
    def process_vacancy(self):
        """
        Тестирование метода process-vacancy
        """
        self.client.post("/process-vacancy/", txt4)
