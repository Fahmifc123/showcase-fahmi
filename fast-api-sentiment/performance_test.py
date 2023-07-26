import json
from locust import HttpUser, task, between

class PerformanceTests(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def testFastApi(self):
        sample = [
            {
                "text": "Mendengar jawaban singkat Presiden, Mensesneg Pratikno, Sekretaris Kabinet Pramono Anung, Pj Gubernur DKI Jakarta Heru Budi Hartono dan Menteri Investasi Bahlil Lahadalia yang mendampingi Jokowi tertawa.",
                "id": "xyz"
            }
        ]
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        self.client.post("/sentiment/v3/predict-id/",
            data=json.dumps(sample), headers=headers, name="Predict ID")

