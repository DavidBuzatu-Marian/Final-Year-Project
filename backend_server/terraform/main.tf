provider "google" {
  project = "finalyearproject-338819"
  region  = "europe-west1"
  zone    = "europe-west1-b"
}

resource "google_compute_instance" "controller_instance" {
  count        = var.nr_controllers
  name         = "controller-${count.index}"
  machine_type = var.machine_type

  boot_disk {
    initialize_params {
      image = var.controller_image
    }
  }
  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("./startup_script.sh")

  service_account {
    scopes = ["cloud-platform", "cloud-source-repos", "compute-rw"]
  }

}


