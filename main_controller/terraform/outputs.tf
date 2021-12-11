output "ec2_instances_id" {
  value = aws_instance.test_server.*.id
}

output "ec2_instances_public_ip" {
  value = aws_instance.test_server.*.public_ip
}
