output "gci_instances_id" {
  value = google_compute_instance.instances[*].network_interface.0.network_ip
}

output "gci_instances_public_ip" {
  value = google_compute_instance.instances[*].network_interface.0.network_ip
}
