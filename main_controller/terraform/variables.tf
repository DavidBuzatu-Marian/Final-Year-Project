variable "nr_instances" {
  type        = number
  description = "The number of instances to create"
  default     = 1

  validation {
    condition     = var.nr_instances >= 1 && var.nr_instances <= 10
    error_message = "The instance number has to be between 1 and 10 (inclusive)."
  }
}

variable "username" {
  type        = string
  description = "The user name for the current environment"
}
