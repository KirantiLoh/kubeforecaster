import random
from locust import FastHttpUser, TaskSet, task
from faker import Faker
import datetime
import time
import logging

fake = Faker()

# Products and endpoints
products = [
    '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP', '6E92ZMYYFZ',
    '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z'
]

currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']
checkout_weights = [0.6, 0.2, 0.1, 0.05, 0.05]  # Weighted randomness

# Exponential delay with jitter
def random_delay():
    delay = random.expovariate(0.5)  # Mean ~2 seconds
    time.sleep(min(delay, 10))       # Cap max delay

# Random task selection with weighted probability
def random_task_choice():
    return random.choices([
        'index',
        'set_currency',
        'browse_product',
        'add_to_cart',
        'view_cart',
        'checkout',
        'empty_cart',
    ], weights=[10, 5, 10, 5, 3, 2, 1, 1], k=1)[0]

class UserBehavior(TaskSet):
    def on_start(self):
        self.client.get("/")

    @task
    def execute_random_task(self):
        task_name = random_task_choice()
        try:
            if task_name == 'index':
                self.client.get("/")
            elif task_name == 'set_currency':
                self.client.post("/setCurrency", {'currency_code': random.choice(currencies)})
            elif task_name == 'browse_product':
                self.client.get(f"/product/{random.choice(products)}")
            elif task_name == 'add_to_cart':
                self.add_to_cart()
            elif task_name == 'view_cart':
                self.client.get("/cart")
            elif task_name == 'checkout':
                self.checkout()
            elif task_name == 'empty_cart':
                self.client.post("/cart/empty")
        except Exception as e:
            logging.error(f"Error during {task_name}: {str(e)}")
        finally:
            random_delay()

    def add_to_cart(self):
        product = random.choice(products)
        quantity = random.randint(1, 10)
        self.client.get(f"/product/{product}")
        self.client.post("/cart", {'product_id': product, 'quantity': quantity})

    def checkout(self):
        # Add multiple items with variable quantities
        for _ in range(random.randint(1, 3)):
            self.add_to_cart()

        current_year = datetime.datetime.now().year + 1
        self.client.post("/cart/checkout", {
            'email': fake.email(),
            'street_address': fake.street_address(),
            'zip_code': fake.zipcode(),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'country': fake.country(),
            'credit_card_number': fake.credit_card_number(card_type="visa"),
            'credit_card_expiration_month': random.randint(1, 12),
            'credit_card_expiration_year': random.randint(current_year, current_year + 70),
            'credit_card_cvv': f"{random.randint(100, 999)}",
        })

class WebsiteUser(FastHttpUser):
    tasks = [UserBehavior]
    wait_time = lambda: random.uniform(0.5, 10) 