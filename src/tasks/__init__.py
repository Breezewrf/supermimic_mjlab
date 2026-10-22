"""Task registration is explicit so tensor/data utilities work without a simulator."""


def register_tasks():
  from mjlab.utils.lab_api.tasks.importer import import_packages

  import_packages(__name__, ["utils", ".mdp", "supermimic"])
