output "ec2_instance_id" {
  value = aws_instance.test_server.id
}

output "ec2_instance_public_ip" {
  value = aws_instance.test_server.public_ip
}
