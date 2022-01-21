provider "google" {
  project = "finalyearproject-338819"
  region  = "europe-west1"
  zone    = "europe-west1-b"
}

resource "google_compute_instance" "vm_instance" {
  name         = "terraform-test-instance"
  machine_type = "e2-micro"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"
    }
  }
  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("./startup_script.sh")
}
