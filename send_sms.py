# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
account_sid = 'AC9a2390a48b67fd3b749b74bd5b86ed28'
auth_token = 'c9196a9d80b28804445ca70276d507c4'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Join Earth's mightiest heroes. Like Kevin Bacon.",
                     from_='+16507536403',
                     to='+16508673228'
                 )

print(message.sid)