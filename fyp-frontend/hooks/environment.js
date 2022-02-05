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
