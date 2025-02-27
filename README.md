## Mlops Simple GAN Model

## Check List
- Dockerfile
- Github Workflow file
- Github Secrets for workflow
- AWS Iam User for AWS Access Keys
- Create AWS ECR repo for docker image storage and take repo adress it
- AWS EC2 for run docker image

## For Docker Setup In EC2 commands

sudo apt-get update -y

sudo apt-get upgrade

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## GitHub Actions - AWS EC2 options and commands
- go github runners and create linux runners and paste commands to AWS EC2 
- So EC2 is connect our github actions

## If you push anything to github, docker and application in aws will automatically change accordingly.
