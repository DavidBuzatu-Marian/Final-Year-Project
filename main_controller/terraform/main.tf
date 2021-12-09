terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.6"
    }
  }
}

provider "aws" {
  profile                 = "testing"
  region                  = "eu-west-2"
  shared_credentials_file = "./aws_credentials"
}

resource "aws_instance" "test_server" {
  ami           = "ami-0194c3e07668a7e36"
  instance_type = "t2.micro"
  count         = var.instance_number

  tags = {
    Name = "test_instance"
  }
}
