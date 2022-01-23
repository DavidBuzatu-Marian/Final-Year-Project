variable "controller_image" {
  type        = string
  description = "The type of image to use for controllers"
  default     = "projects/finalyearproject-338819/global/images/controller-image"
}

variable "nr_controllers" {
  type        = number
  description = "The number of controllers to create"
  default     = 1

  validation {
    condition     = var.nr_controllers >= 1 && var.nr_controllers <= 8
    error_message = "The controllers number has to be between 1 and 8 (inclusive)."
  }
}

variable "machine_type" {
  type        = string
  description = "The type of machine instance for the controller"
  default     = "e2-medium"
}
