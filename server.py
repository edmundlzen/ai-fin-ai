from flask import Flask, jsonify, request
from ai import calculate

app = Flask(__name__)


@app.route("/calculate-recommendation", methods=["POST"])
def calculate_recommendation():
    data = request.get_json()
    user_investment_amount = data["investment_amount"]
    sigma = data["sigma"]

    if not isinstance(user_investment_amount, (int, float)) or not isinstance(
        sigma, (int, float)
    ):
        print("ValueTypeError")
        response = jsonify({"error": "Investment amount and sigma must be a number"})
        response.status_code = 400
        return response

    if sigma < 0 or sigma > 1:
        print("OutOfRangeError")
        response = jsonify({"error": "Sigma must be between 0 and 1"})
        response.status_code = 400
        return response

    recommendation = calculate(user_investment_amount, sigma)

    response = jsonify(recommendation)
    response.headers.add("Access-Control-Allow-Origin", "*")
    print(recommendation)
    return response


if __name__ == "__main__":
    app.run(debug=True)
