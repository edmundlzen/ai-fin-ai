from faker import Faker
from tabulate import tabulate
import random

fake = Faker()


def generate_fake_fund_data(num_funds):
    funds = []

    for _ in range(num_funds):
        fund = {
            "fund_name": fake.company(),
            "fund_type": random.choice(["Equity", "Fixed Income", "Balanced"]),
            "asset_under_management": round(random.uniform(1000000, 1000000000), 2),
            "annual_return": round(random.uniform(0.05, 0.15), 4),
            "volatility": round(random.uniform(0.05, 0.15), 4),
            "manager": fake.name(),
            "inception_date": fake.date_this_decade(),
        }
        funds.append(fund)

    return funds


# Generate a list of 10 fake unit trust funds
# num_funds = 10
# fake_fund_data = generate_fake_fund_data(num_funds)
