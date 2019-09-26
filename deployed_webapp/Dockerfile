# Creates a docker image from the working image webapp_ubuntu_complete that runs flask on start
FROM webapp_ubuntu_complete:latest

MAINTAINER Matt Reese <mwreese@ucdavis.edu>

WORKDIR /usr/src/webapp

EXPOSE 5000

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=run.py

CMD ["flask", "run", "--host", "0.0.0.0"]
