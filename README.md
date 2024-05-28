# Object detection

A local http server that accepts POST request at `/api/v1/users`. On receiving
the request, the server:
1. Send a request to StackExchange to fetch top 10 users' profiles
   on StackOverflow, ordered by highest reputation
2. For each of the top 10 users, read its profile image, fetch the image,
   and use a object detection model to get a list of bounding boxes
3. The request includes an object type, eg. 'person', then search bounding
   boxes, to find bounding boxes with the same lable as the object type
   specified in the request
4. Return the list of users, with their information, and detection results as
   JSON payload.

To run the server, and test it with `curl`:
```
pip3 install -r requirements.txt
python3 od.py
curl -X POST http://127.0.0.1:5000/api/v1/users \
    -H "Content-Type: application/json" \
    -d '{"query": {"object":"person"}}'
```

To run the unit tests:
```
python3 -m unittest od_test.py
```

NOTE: I have python3, and do not use python alias, you may need to change to use
`python`.

## TODO

1. Enable streaming: for each stackoverflow users user, once finished streaming
   back the results to client.
2. Enable specifying arbitrary number of users. Client can specify to fetch `n`
   users. And server needs to use pagination when fetching user profiles,
   in order to ensure smooth processing.

## Notes

This has been done in the span of one day. The actual working time is about 3
hours.
