output "gci_controller_ids" {
  value = google_compute_instance.controller_instance[*].network_interface.0.network_ip
}
