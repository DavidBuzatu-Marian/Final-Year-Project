provider "google" {
  credentials = file("finalyearproject-338819-12b837ed8475.json")
  project     = "finalyearproject-338819"
  region      = "europe-west1"
  zone        = "europe-west1-b"
}


resource "google_compute_instance" "instances" {
  count        = var.nr_instances
  name         = "instance-${var.user_id}${count.index}"
  machine_type = var.machine_type

  boot_disk {
    initialize_params {
      image = var.instance_image
    }
  }
  network_interface {
    network = "default"
    access_config {

    }
  }

  metadata_startup_script = file("./startup_script.sh")

  service_account {
    scopes = ["cloud-platform", "cloud-source-repos", "compute-rw"]
  }
}

# resource "google_compute_instance_group" "environment" {
#   name        = "instance-group-${var.user_id}"
#   description = "Environment group. Only 1 environment/user"
#   instances   = google_compute_instance.instances[*].self_link

#   named_port {
#     name = "http"
#     port = "5000"
#   }
# }
