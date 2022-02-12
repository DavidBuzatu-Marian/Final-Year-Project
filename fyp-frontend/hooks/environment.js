import { getConfig } from '../config/defaultConfig';
import useSWR from 'swr';
import { fetcher } from './user';

export const useEnvironment = () => {
  const { data, mutate, error } = useSWR(
    getConfig()['environmentAddressesUrl'],
    fetcher
  );
  const loading = !data && !error;
  const environments = data && !error ? data : null;
  if (environments) {
    environments = environments.map((environment) => {
      return { ...environment, id: environment._id };
    });
  }
  return [environments, { mutate, loading }, error];
};

export const useEnvironmentState = (shouldFetch, environmentCreateTaskLink) => {
  const { data, mutate, error } = useSWR(
    shouldFetch ? environmentCreateTaskLink : null,
    fetcher
  );
  const loading = !data && !error;
  const completed = data && !error ? data : null;
  return [completed, { mutate, loading }, error];
};
