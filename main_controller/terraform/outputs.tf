output "gci_instances_ids" {
  value = google_compute_instance.instances[*].network_interface.0.network_ip
}
